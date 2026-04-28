#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) * 4 // 3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PYTHON = "/home/kec23008/.venv/bin/python3"

MODEL_CONFIGS = {
    "qwen32b": {
        "path": "/home/kec23008/Models/Qwen3-32B",
        "tp": 2, "dp": 1, "maxlen": 32768, "mem_frac": "0.85", "cuda_devices": "0,1",
        "max_workers": 8,
        "kv_bytes_per_token": 64 * 2 * 8 * 128 * 2,
    },
    "qwen30b": {
        "path": "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507",
        "tp": 2, "dp": 1, "maxlen": 32768, "mem_frac": "0.90", "cuda_devices": "0,1",
        "max_workers": 12,
        "kv_bytes_per_token": 48 * 2 * 8 * 128 * 2,
    },
    "phi4": {
        "path": "/home/kec23008/Models/Phi-4",
        "tp": 1, "dp": 2, "maxlen": 16384, "mem_frac": "0.90", "cuda_devices": "0,1",
        "max_workers": 24,
        "kv_bytes_per_token": 32 * 2 * 32 * 96 * 2,
    },
}

MODES: Dict[str, dict] = {
    "baseline": {
        "description": "Standard sglang — no SafeKV, full KV cache sharing",
        "server_safekv_args": [],
        "use_user_id": False,
    },
    "private_default": {
        "description": "SafeKV private-by-default — no cross-tenant sharing",
        "server_safekv_args": [
            "--safekv-access-budget", "10",
            "--safekv-creator-threshold", "2",
            "--safekv-private-only",
        ],
        "use_user_id": True,
    },
    "private_detector": {
        "description": "SafeKV + detector — non-PII promoted (B=999999)",
        "server_safekv_args": [
            "--safekv-access-budget", "999999",
            "--safekv-creator-threshold", "1",
        ],
        "use_user_id": True,
    },
    "full_safekv": {
        "description": "Full SafeKV — B=10, K=2",
        "server_safekv_args": [
            "--safekv-access-budget", "10",
            "--safekv-creator-threshold", "2",
        ],
        "use_user_id": True,
    },
}

PII_DATASET = Path("/home/kec23008/InferShield/datasets/english_pii_43k.jsonl")
SHAREGPT_DATASET = Path("/home/kec23008/InferShield/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
TARGET_TOKENS = 4096
COMPLETION_CONFIG = {"max_tokens": 128, "temperature": 0}
# Qwen3 MoE (qwen30b) uses thinking mode by default; suppress it to get
# accurate TTFT and output-token counts.
MODEL_COMPLETION_OVERRIDES: dict = {
    "qwen30b": {"enable_thinking": False},
    "qwen32b": {"enable_thinking": False},
    "phi4":    {"enable_thinking": False},
}
METRICS_BATCH = 50

# Dataset design: "shared context + private PII suffix"
#   SHARED_CONTEXT_TOKENS tokens of non-PII text — identical for ALL N users
#   SUFFIX_TOKENS tokens of unique content per user (PII or non-PII)
#
# Expected cumul_kv_tokens per mode:
#   baseline        : shared once + N × suffix  ≈ N × SUFFIX_TOKENS   (prefix reused)
#   private_default : everything isolated        ≈ N × TARGET_TOKENS   (~2x higher)
#   full_safekv     : prefix detected non-PII → shared; PII suffix isolated
#                     ≈ same as baseline         (SafeKV retains prefix efficiency)
SHARED_CONTEXT_TOKENS = 2048  # non-PII shared prefix length


@dataclass
class ReqResult:
    request_id: int
    user_id: Optional[str]
    is_pii: bool
    ttft: float
    tpop: float
    n_output_tokens: int
    total_time: float
    error: Optional[str] = None


@dataclass
class ModeResult:
    mode: str
    description: str
    req_results: List[ReqResult] = field(default_factory=list)
    kv_timeline: List[Tuple[float, int]] = field(default_factory=list)
    wall_time: float = 0.0
    ttft_mean_pii: float = float("nan")
    ttft_p50_pii: float = float("nan")
    ttft_p95_pii: float = float("nan")
    ttft_p99_pii: float = float("nan")
    ttft_mean_nonpii: float = float("nan")
    ttft_p50_nonpii: float = float("nan")
    ttft_p95_nonpii: float = float("nan")
    ttft_p99_nonpii: float = float("nan")
    tpop_mean: float = float("nan")
    tpop_p50: float = float("nan")
    tpop_p95: float = float("nan")
    throughput_toks: float = 0.0
    cumul_kv_tokens: int = 0
    cumul_kv_gb: float = 0.0


def _kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL)
        pids = out.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid], capture_output=True, check=False)
                print(f"  [kill] PID {pid} on :{port}", flush=True)
        if pids:
            time.sleep(4)
    except subprocess.CalledProcessError:
        pass


def _start_server(model: str, port: int, safekv_args: List[str], log_tag: str) -> subprocess.Popen:
    cfg = MODEL_CONFIGS[model]
    log_path = LOG_DIR / f"throughput_{model}_{log_tag}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]
    env["LD_LIBRARY_PATH"] = (
        "/home/kec23008/.local/lib/python3.10/site-packages/nvidia/nvshmem/lib:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path",          cfg["path"],
        "--host",                "127.0.0.1",
        "--port",                str(port),
        "--dtype",               "float16",
        "--trust-remote-code",
        "--tp-size",             str(cfg["tp"]),
        "--dp-size",             str(cfg["dp"]),
        "--context-length",      str(cfg["maxlen"]),
        "--served-model-name",   model,
        "--attention-backend",   "torch_native",
        "--disable-cuda-graph",
        "--mem-fraction-static", cfg["mem_frac"],
        "--enable-metrics",
    ] + cfg.get("extra_server_args", []) + safekv_args

    print(f"  [start] {' '.join(cmd[:6])} ...", flush=True)
    print(f"  [log]   {log_path}", flush=True)

    log_fh = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env)


def _wait_ready(port: int, timeout: int = 420) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    spinner = ["|", "/", "-", "\\"]
    i = 0
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"\n  [ready] Server on :{port} is up!", flush=True)
                return True
        except Exception:
            pass
        print(f"\r  [wait] {spinner[i % 4]} polling :{port} ...", end="", flush=True)
        i += 1
        time.sleep(5)
    print(f"\n  [ERROR] Server on :{port} did not start within {timeout}s", flush=True)
    return False


def _send_one(port: int, model: str, req_id: int, user_id: Optional[str],
              prompt: str, is_pii: bool) -> ReqResult:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    extra = MODEL_COMPLETION_OVERRIDES.get(model, {})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        **COMPLETION_CONFIG,
        **extra,
    }
    if user_id is not None:
        payload["user_id"] = user_id

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    n_tokens = 0
    error: Optional[str] = None

    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                             stream=True, timeout=300)
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                content = (chunk.get("choices", [{}])[0].get("delta", {}).get("content", ""))
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_tokens += 1
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - t0
    if ttft is None:
        ttft = elapsed

    n_out = max(n_tokens, 1)
    tpop = (elapsed - ttft) / n_out if n_out > 1 else 0.0

    return ReqResult(
        request_id=req_id,
        user_id=user_id,
        is_pii=is_pii,
        ttft=ttft,
        tpop=tpop,
        n_output_tokens=n_out,
        total_time=elapsed,
        error=error,
    )


def _get_metrics(port: int, model: str) -> Tuple[int, int]:
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=10)
        resp.raise_for_status()
        text = resp.text
        prompt_tokens = 0
        cached_tokens = 0
        model_label = model
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if f'sglang:prompt_tokens_total{{model_name="{model_label}"}}' in line:
                try:
                    prompt_tokens = int(float(line.split()[-1]))
                except (ValueError, IndexError):
                    pass
            elif f'sglang:cached_tokens_total{{model_name="{model_label}"}}' in line:
                try:
                    cached_tokens = int(float(line.split()[-1]))
                except (ValueError, IndexError):
                    pass
        return prompt_tokens, cached_tokens
    except Exception as exc:
        print(f"  [WARN] metrics error: {exc}", flush=True)
        return 0, 0


def build_dataset(n: int = 1000) -> List[Tuple[str, str, bool]]:
    """
    "Shared context + private PII suffix" workload design:

      Every request = SHARED_CONTEXT (non-PII, ~SHARED_CONTEXT_TOKENS tokens,
                                      identical for ALL N users)
                    + UNIQUE_SUFFIX  (unique per user, ~suffix_tokens tokens)

    n//2 requests are PII (unique PII suffix, user_id = "pii_user_{i}")
    n//2 requests are non-PII (unique non-PII suffix, user_id = "reg_user_{i}")

    Expected cumul_kv_tokens:
      baseline / private_detector / full_safekv:
          ≈ SHARED_CONTEXT_TOKENS  +  n × suffix_tokens
          (shared context stored once; unique suffixes stored once each)
      private_default:
          ≈ n × TARGET_TOKENS
          (everything isolated; shared context re-prefilled for every user)
      Ratio ≈ TARGET_TOKENS / suffix_tokens  ≈  4096 / 2048 ≈ 2×

    This ensures ≥ 50 % of each request's tokens are reusable non-PII content,
    clearly exceeding the 20 % reuse target.
    """
    # suffix_tokens computed after shared_context is built (see below)
    n_pii   = n          # all requests are PII
    n_nonpii = 0
    print(f"  [dataset] Building {n} requests (ALL PII)  "
          f"(shared={SHARED_CONTEXT_TOKENS}t non-PII prefix + "
          f"suffix=~{TARGET_TOKENS - SHARED_CONTEXT_TOKENS}t unique PII per user) ...",
          flush=True)

    # ── load raw corpora ──────────────────────────────────────────────────────
    pii_entries: List[str] = []
    with open(PII_DATASET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("source_text", "")
                if text:
                    pii_entries.append(text)
            except json.JSONDecodeError:
                pass
    print(f"  [dataset] Loaded {len(pii_entries)} PII entries", flush=True)

    sharegpt_turns: List[str] = []
    with open(SHAREGPT_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        for conv in item.get("conversations", []):
            if conv.get("from") == "human":
                text = conv.get("value", "").strip()
                if text:
                    sharegpt_turns.append(text)
    print(f"  [dataset] Loaded {len(sharegpt_turns)} ShareGPT human turns", flush=True)

    random.shuffle(pii_entries)
    random.shuffle(sharegpt_turns)

    # ── build ONE shared non-PII context (same for every request) ────────────
    # Use a FIXED, verified PII-free clinical reference document.
    # Random ShareGPT turns are excluded because they may contain personal
    # information (names, emails, dates) that the PII detector correctly flags,
    # preventing promotion and making the experiment invalid.
    FIXED_SHARED_CONTEXT = """\
Clinical Decision Support Reference Manual — Version 4.2

SECTION 1: INTRODUCTION TO EVIDENCE-BASED CLINICAL GUIDELINES

Evidence-based medicine integrates the best available research evidence with clinical expertise and patient values. Clinical decision support systems (CDSS) are designed to assist healthcare providers in applying these guidelines at the point of care. This manual describes the standard protocols used by integrated health networks when processing patient encounter data, diagnostic codes, and treatment pathway recommendations.

SECTION 2: DIAGNOSTIC CODING STANDARDS

The International Classification of Diseases, Tenth Revision, Clinical Modification (ICD-10-CM) provides a standardized vocabulary for encoding diagnoses, symptoms, and medical conditions. Each code consists of an alphanumeric prefix followed by numeric subcategories that specify the condition, anatomical site, severity, and encounter type. For example, codes in the range J00-J99 classify diseases of the respiratory system, while codes M00-M99 address musculoskeletal disorders and connective tissue diseases.

Healthcare providers must apply the principal diagnosis code—the condition established after study to be chiefly responsible for the admission—along with secondary codes for comorbidities that affect patient management or resource utilization. Accurate coding directly influences reimbursement rates under the Diagnosis-Related Group (DRG) payment system and population health analytics used for quality improvement programs.

SECTION 3: LABORATORY VALUE INTERPRETATION THRESHOLDS

Standard reference intervals represent the range of values observed in a healthy reference population. Clinical laboratories establish these intervals through statistical analysis of measurement distributions and calibrate instruments according to Clinical Laboratory Improvement Amendments (CLIA) regulations.

Common metabolic panel reference ranges include: sodium (136-145 mEq/L), potassium (3.5-5.0 mEq/L), chloride (98-107 mEq/L), bicarbonate (22-29 mEq/L), blood urea nitrogen (7-25 mg/dL), creatinine (0.6-1.2 mg/dL for adults), glucose (70-100 mg/dL fasting), and calcium (8.5-10.5 mg/dL). Values outside these ranges trigger alert notifications within the CDSS, which cross-references current medication lists and active diagnoses to generate differential considerations and recommended follow-up orders.

Complete blood count parameters include hemoglobin (12.0-17.5 g/dL), hematocrit (36-50%), white blood cell count (4.5-11.0 × 10³/μL), and platelet count (150-400 × 10³/μL). The differential white cell count provides further granularity: neutrophils (50-70%), lymphocytes (20-40%), monocytes (2-8%), eosinophils (1-4%), and basophils (0.5-1%).

SECTION 4: MEDICATION RECONCILIATION PROTOCOLS

Medication reconciliation is the formal process of comparing a patient's medication orders to all medications the patient has been taking. This reconciliation is required at each care transition—admission, transfer between units, and discharge—to identify and resolve discrepancies such as omissions, duplications, dosing errors, and drug-drug interactions.

The reconciliation workflow proceeds through four phases: (1) collection of a complete and accurate medication list from all available sources; (2) comparison of this list against newly prescribed orders; (3) resolution of identified discrepancies with appropriate clinical justification; and (4) communication of the updated medication list to the receiving care team and to the patient.

Drug interaction databases such as Micromedex and Clinical Pharmacology classify interactions by severity (contraindicated, major, moderate, minor) and documentation status (excellent, good, fair). The CDSS automatically queries these databases when new medications are ordered and presents relevant interaction alerts to the prescribing clinician with recommended management options.

SECTION 5: VITAL SIGN TRENDING AND EARLY WARNING SCORES

Vital sign monitoring provides continuous assessment of physiological status. Standard parameters include heart rate (normal range 60-100 beats per minute), respiratory rate (12-20 breaths per minute), blood pressure (systolic 90-140 mmHg, diastolic 60-90 mmHg), oxygen saturation (≥95% on room air), and body temperature (36.1-37.2°C or 97.0-99.0°F).

The Modified Early Warning Score (MEWS) aggregates vital sign deviations into a composite score that predicts clinical deterioration. Each parameter receives a weighted score from 0 to 3 based on its deviation from normal: a MEWS of 4 or greater prompts escalation to a rapid response team evaluation. Automated MEWS calculation within the electronic health record (EHR) alerts bedside nurses and charge nurses when threshold criteria are met.

SECTION 6: CLINICAL DOCUMENTATION STANDARDS

Structured clinical documentation improves care coordination, reduces redundancy, and enables secondary use of data for quality measurement and research. The SOAP note format—Subjective, Objective, Assessment, and Plan—provides a standardized framework for clinical encounter documentation. Subjective data encompasses the patient's reported symptoms and history; objective data includes examination findings, vital signs, and diagnostic results; the assessment synthesizes this information into a differential diagnosis; and the plan outlines therapeutic interventions, follow-up instructions, and referrals.

Template-driven documentation tools incorporate mandatory fields aligned with regulatory requirements and accreditation standards from organizations such as The Joint Commission and the Centers for Medicare and Medicaid Services. Completion rates and documentation quality metrics are monitored through the quality management dashboard and reported to department heads on a monthly basis.

SECTION 7: INFECTION CONTROL AND ANTIMICROBIAL STEWARDSHIP

Antimicrobial stewardship programs (ASP) optimize antibiotic selection, dosing, route, and duration to minimize adverse effects, reduce rates of resistance, and improve patient outcomes. The ASP team reviews culture and sensitivity results, pharmacokinetic parameters, and treatment response indicators to provide evidence-based recommendations to prescribers.

Common antimicrobial resistance mechanisms include beta-lactamase production, target site modification, efflux pump overexpression, and outer membrane protein loss. Extended-spectrum beta-lactamase (ESBL)-producing organisms require carbapenem therapy in most clinical scenarios. Carbapenem-resistant Enterobacteriaceae (CRE) mandate contact precautions, isolation procedures, and notification to the local public health authority in accordance with reportable disease regulations.

SECTION 8: QUALITY METRICS AND PERFORMANCE INDICATORS

Core quality measures reported to national registries include door-to-balloon time for ST-elevation myocardial infarction (target: ≤90 minutes), pneumonia vaccination rates, surgical site infection rates, catheter-associated urinary tract infection rates per 1000 catheter-days, and central line-associated bloodstream infection rates per 1000 central line-days.

The Leapfrog Group, National Quality Forum, and Agency for Healthcare Research and Quality publish composite quality scores that aggregate these measures into summary statistics used by payers, employers, and patients when evaluating healthcare organization performance. Internal benchmarking compares unit-level metrics against peer institutions stratified by bed size, teaching status, and patient complexity.

SECTION 9: HEALTH INFORMATION EXCHANGE AND INTEROPERABILITY

Health information exchange (HIE) enables the electronic movement of clinical information among disparate healthcare organizations according to nationally recognized standards. The Fast Healthcare Interoperability Resources (FHIR) standard, developed by Health Level Seven International (HL7), provides a framework for exchanging healthcare information electronically using modern web technologies including RESTful APIs, JSON, and XML data formats.

FHIR resources represent discrete clinical concepts such as Patient, Observation, Condition, MedicationRequest, and DiagnosticReport. Each resource contains structured data elements mapped to standardized terminologies: SNOMED CT for clinical findings and procedures, LOINC for laboratory and clinical observations, RxNorm for medications, and ICD-10 for diagnoses. Terminology binding ensures semantic interoperability when data moves between systems with different internal representations.

The 21st Century Cures Act mandates that certified health IT developers support application programming interfaces without special effort, ensuring that patients and providers can access, exchange, and use electronic health information. Certified APIs must conform to the United States Core Data for Interoperability (USCDI) data set and support the HL7 FHIR Release 4 standard as specified in the ONC Interoperability and Information Blocking regulation.

SECTION 10: PATIENT SAFETY AND RISK MANAGEMENT

The Institute for Healthcare Improvement champions the Triple Aim framework—improving population health, enhancing the patient experience, and reducing per capita cost—as the foundation for healthcare system redesign. Patient safety initiatives targeting preventable harm employ root cause analysis, failure mode and effects analysis, and prospective risk assessment methodologies to identify systemic vulnerabilities before adverse events occur.

Closed-loop medication administration combines barcode medication administration scanning, automated dispensing cabinet integration, and electronic medication administration record documentation to verify the five rights of medication administration: right patient, right drug, right dose, right route, and right time. Near-miss reporting systems capture safety events that did not result in harm but reveal latent system weaknesses requiring corrective action.
"""
    shared_context = FIXED_SHARED_CONTEXT.strip()
    actual_shared_tokens = count_tokens(shared_context)
    suffix_tokens = max(512, TARGET_TOKENS - actual_shared_tokens)
    print(f"  [dataset] Shared context: {actual_shared_tokens} tokens (fixed PII-free clinical document)", flush=True)
    print(f"  [dataset] Suffix target: {suffix_tokens} tokens per request", flush=True)

    # ── helper: build a unique suffix of ~suffix_tokens ──────────────────────
    def _build_suffix(entries: List[str], idx: int) -> Tuple[str, int]:
        parts: List[str] = []
        total = 0
        while total < suffix_tokens and idx < len(entries):
            t = entries[idx]; idx += 1
            parts.append(t)
            total += count_tokens(t)
        if idx >= len(entries):
            idx = 0
        text = "\n\n".join(parts)
        if count_tokens(text) < suffix_tokens // 2:
            text += "\n\n" + entries[idx % len(entries)]; idx += 1
        return text, idx

    # ── PII requests: shared_context + unique PII suffix ─────────────────────
    dataset: List[Tuple[str, str, bool]] = []
    pii_idx = 0
    pii_section_header = (
        "\n\n---\nConfidential patient / user record (handle with care):\n\n"
    )
    pii_task = (
        "\n\n---\nTask: Based on the above record, identify the key personal "
        "information fields present (e.g., name, date of birth, medical conditions, "
        "financial details) and briefly assess any privacy risks associated with "
        "this data. Provide a structured 3-5 sentence summary."
    )
    for i in range(n_pii):
        suffix, pii_idx = _build_suffix(pii_entries, pii_idx)
        prompt = shared_context + pii_section_header + suffix + pii_task
        uid = f"pii_user_{i}"
        dataset.append((prompt, uid, True))
        if i % 100 == 0:
            print(f"    PII [{i}/{n_pii}] total_tokens={count_tokens(prompt)}", flush=True)

    random.shuffle(dataset)
    reuse_pct = 100 * SHARED_CONTEXT_TOKENS // TARGET_TOKENS
    print(f"  [dataset] Total: {len(dataset)} PII requests  "
          f"(shared non-PII prefix = {SHARED_CONTEXT_TOKENS}/{TARGET_TOKENS} "
          f"= {reuse_pct}% of each request, reusable across all users)",
          flush=True)
    return dataset


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_mode(mode_name: str, mode_cfg: dict, port: int, model: str,
             dataset: List[Tuple[str, str, bool]], max_workers: int) -> ModeResult:
    use_uid = mode_cfg["use_user_id"]
    desc = mode_cfg["description"]
    result = ModeResult(mode=mode_name, description=desc)

    print(f"\n{'─'*60}", flush=True)
    print(f"  Mode: {mode_name}", flush=True)
    print(f"  {desc}", flush=True)
    print(f"  max_workers={max_workers}  n_requests={len(dataset)}", flush=True)
    print(f"{'─'*60}", flush=True)

    print(f"  [warmup] Sending 5 warmup requests...", flush=True)
    warmup_prompt = "Hello, what is 2 + 2?"
    for j in range(5):
        try:
            _send_one(port, model, j, "warmup_user" if use_uid else None,
                      warmup_prompt, False)
        except Exception as e:
            print(f"    [WARN] warmup error: {e}", flush=True)
    print(f"  [warmup] Done", flush=True)
    time.sleep(2)

    base_pt, base_ct = _get_metrics(port, model)
    print(f"  [metrics] baseline prompt_tokens={base_pt}  cached_tokens={base_ct}", flush=True)

    t_wall_start = time.time()
    completed_count = 0
    kv_timeline: List[Tuple[float, int]] = []
    cumul_new_tokens = 0

    def task(i: int, prompt: str, uid: Optional[str], is_pii: bool) -> ReqResult:
        effective_uid = uid if use_uid else None
        return _send_one(port, model, i, effective_uid, prompt, is_pii)

    req_results: List[ReqResult] = []
    server_dead = False
    consec_metric_fails = 0

    def _server_alive() -> bool:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(task, i, prompt, uid, is_pii): i
            for i, (prompt, uid, is_pii) in enumerate(dataset)
        }
        for fut in as_completed(futs):
            if server_dead:
                # cancel remaining futures — they'll all fail anyway
                fut.cancel()
                continue
            try:
                r = fut.result()
                req_results.append(r)
            except Exception as e:
                req_id = futs[fut]
                req_results.append(ReqResult(
                    request_id=req_id, user_id=None, is_pii=False,
                    ttft=0.0, tpop=0.0, n_output_tokens=0,
                    total_time=0.0, error=str(e)
                ))

            completed_count += 1
            if completed_count % METRICS_BATCH == 0 or completed_count == len(dataset):
                t_now = time.time() - t_wall_start
                try:
                    cur_pt, cur_ct = _get_metrics(port, model)
                    delta_prompt = cur_pt - base_pt
                    delta_cached = cur_ct - base_ct
                    new_kv = max(0, delta_prompt - delta_cached)
                    cumul_new_tokens = new_kv
                    kv_timeline.append((t_now, new_kv))
                    consec_metric_fails = 0
                    print(f"  [{completed_count}/{len(dataset)}] t={t_now:.1f}s  cumul_kv_tokens={new_kv}", flush=True)
                except Exception as e:
                    consec_metric_fails += 1
                    print(f"  [WARN] metrics error #{consec_metric_fails}: {e}", flush=True)
                    if consec_metric_fails >= 3 and not _server_alive():
                        print(f"  [ERROR] Server on :{port} is dead — aborting mode early.", flush=True)
                        server_dead = True

    result.wall_time = time.time() - t_wall_start
    result.req_results = req_results
    result.kv_timeline = kv_timeline
    result.cumul_kv_tokens = cumul_new_tokens

    kv_bytes = MODEL_CONFIGS[model]["kv_bytes_per_token"]
    result.cumul_kv_gb = cumul_new_tokens * kv_bytes / (1024 ** 3)

    ok = [r for r in req_results if r.error is None and r.n_output_tokens > 0]
    pii_ok  = sorted([r.ttft for r in ok if r.is_pii])
    npii_ok = sorted([r.ttft for r in ok if not r.is_pii])
    tpop_ok = sorted([r.tpop for r in ok if r.n_output_tokens > 1])

    if pii_ok:
        import statistics as _stats
        result.ttft_mean_pii = _stats.mean(pii_ok)
        result.ttft_p50_pii  = percentile(pii_ok, 0.50)
        result.ttft_p95_pii  = percentile(pii_ok, 0.95)
        result.ttft_p99_pii  = percentile(pii_ok, 0.99)

    if npii_ok:
        import statistics as _stats
        result.ttft_mean_nonpii = _stats.mean(npii_ok)
        result.ttft_p50_nonpii  = percentile(npii_ok, 0.50)
        result.ttft_p95_nonpii  = percentile(npii_ok, 0.95)
        result.ttft_p99_nonpii  = percentile(npii_ok, 0.99)

    if tpop_ok:
        import statistics as _stats
        result.tpop_mean = _stats.mean(tpop_ok)
        result.tpop_p50  = percentile(tpop_ok, 0.50)
        result.tpop_p95  = percentile(tpop_ok, 0.95)

    total_output_toks = sum(r.n_output_tokens for r in ok)
    result.throughput_toks = total_output_toks / result.wall_time if result.wall_time > 0 else 0.0

    print(f"\n  ── {mode_name} summary ──", flush=True)
    print(f"  wall_time={result.wall_time:.1f}s  ok={len(ok)}/{len(req_results)}", flush=True)
    print(f"  TTFT PII  mean/p50/p95/p99: {result.ttft_mean_pii:.3f}/{result.ttft_p50_pii:.3f}/{result.ttft_p95_pii:.3f}/{result.ttft_p99_pii:.3f} s", flush=True)
    print(f"  TTFT Non  mean/p50/p95/p99: {result.ttft_mean_nonpii:.3f}/{result.ttft_p50_nonpii:.3f}/{result.ttft_p95_nonpii:.3f}/{result.ttft_p99_nonpii:.3f} s", flush=True)
    print(f"  TPOP mean/p50/p95: {result.tpop_mean:.4f}/{result.tpop_p50:.4f}/{result.tpop_p95:.4f} s/tok", flush=True)
    print(f"  Throughput: {result.throughput_toks:.2f} tok/s", flush=True)
    print(f"  Cumul KV cache: {result.cumul_kv_tokens} tokens  {result.cumul_kv_gb:.2f} GB", flush=True)

    return result


def print_table(results: List[ModeResult]) -> None:
    sep = "=" * 120
    print(f"\n{sep}")
    print("  SafeKV Throughput Ablation — Summary")
    print(sep)
    hdr = (
        f"{'Mode':<20} "
        f"{'TTFT_pii_mean':>14} {'TTFT_pii_p50':>12} {'TTFT_pii_p95':>12} "
        f"{'TTFT_nop_mean':>14} {'TTFT_nop_p50':>12} "
        f"{'TPOP_mean':>10} "
        f"{'Throughput':>11} "
        f"{'KV_tokens':>12} "
        f"{'KV_GB':>8}"
    )
    print(hdr)
    print("─" * 120)
    for r in results:
        row = (
            f"{r.mode:<20} "
            f"{r.ttft_mean_pii:>14.3f} {r.ttft_p50_pii:>12.3f} {r.ttft_p95_pii:>12.3f} "
            f"{r.ttft_mean_nonpii:>14.3f} {r.ttft_p50_nonpii:>12.3f} "
            f"{r.tpop_mean:>10.4f} "
            f"{r.throughput_toks:>11.2f} "
            f"{r.cumul_kv_tokens:>12} "
            f"{r.cumul_kv_gb:>8.2f}"
        )
        print(row)
    print(sep)


def save_csv(results: List[ModeResult], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "mode", "description",
            "ttft_mean_pii", "ttft_p50_pii", "ttft_p95_pii",
            "ttft_mean_nonpii", "ttft_p50_nonpii", "ttft_p95_nonpii",
            "tpop_mean", "tpop_p50",
            "throughput_toks", "cumul_kv_tokens", "cumul_kv_gb",
        ])
        for r in results:
            writer.writerow([
                r.mode, r.description,
                f"{r.ttft_mean_pii:.4f}", f"{r.ttft_p50_pii:.4f}", f"{r.ttft_p95_pii:.4f}",
                f"{r.ttft_mean_nonpii:.4f}", f"{r.ttft_p50_nonpii:.4f}", f"{r.ttft_p95_nonpii:.4f}",
                f"{r.tpop_mean:.5f}", f"{r.tpop_p50:.5f}",
                f"{r.throughput_toks:.2f}", r.cumul_kv_tokens, f"{r.cumul_kv_gb:.3f}",
            ])
    print(f"  [CSV] Saved: {path}", flush=True)


def save_detail_csv(results: List[ModeResult], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "mode", "request_id", "user_id", "is_pii",
            "ttft", "tpop", "n_output_tokens", "total_time", "error",
        ])
        for r in results:
            for req in r.req_results:
                writer.writerow([
                    r.mode, req.request_id, req.user_id, req.is_pii,
                    f"{req.ttft:.4f}", f"{req.tpop:.5f}", req.n_output_tokens,
                    f"{req.total_time:.4f}", req.error or "",
                ])
    print(f"  [CSV detail] Saved: {path}", flush=True)


def plot_kv_timeline(results: List[ModeResult], path: Path, model: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"baseline": "#d62728", "private_default": "#2ca02c",
              "private_detector": "#1f77b4", "full_safekv": "#ff7f0e"}
    for r in results:
        if not r.kv_timeline:
            continue
        ts = [p[0] for p in r.kv_timeline]
        kv = [p[1] for p in r.kv_timeline]
        ax.plot(ts, kv, label=r.mode, color=colors.get(r.mode, None),
                linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative new KV cache tokens")
    ax.set_title(f"KV Cache Growth — {model}\n(new_kv = delta(prompt_tokens) - delta(cached_tokens))")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [PNG] Saved: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="SafeKV throughput ablation — 4 modes")
    parser.add_argument("--model", default="qwen32b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--n-requests", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=list(MODES.keys()),
                        help="Run only these modes (default: all)")
    parser.add_argument("--budget", type=int, default=None,
                        help="Override access budget B in full_safekv mode")
    args = parser.parse_args()

    # Apply budget override to full_safekv mode before any run
    if args.budget is not None:
        args_list = MODES["full_safekv"]["server_safekv_args"]
        idx = args_list.index("--safekv-access-budget")
        args_list[idx + 1] = str(args.budget)

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_B{args.budget}" if args.budget is not None else ""
    mw_tag = f"_c{args.max_workers}" if args.max_workers is not None else ""
    base_path = Path(args.output) if args.output else LOG_DIR / f"throughput_ablation_{args.model}{tag}{mw_tag}_{ts_str}"
    csv_path = Path(str(base_path) + ".csv") if not str(base_path).endswith(".csv") else base_path
    detail_path = Path(str(base_path).replace(".csv", "") + "_detail.csv")
    png_path = Path(str(base_path).replace(".csv", "") + "_kvcache.png")

    mw = args.max_workers or MODEL_CONFIGS[args.model]["max_workers"]

    print(f"\n{'='*70}", flush=True)
    print(f"  SafeKV Throughput Ablation", flush=True)
    print(f"  Model: {args.model}  Port: {args.port}", flush=True)
    print(f"  n_requests={args.n_requests}  max_workers={mw}", flush=True)
    if args.budget is not None:
        print(f"  budget_override={args.budget}", flush=True)
    print(f"{'='*70}\n", flush=True)

    dataset = build_dataset(n=args.n_requests)

    all_results: List[ModeResult] = []
    modes_to_run = args.modes if args.modes else list(MODES.keys())

    for mode_name in modes_to_run:
        mode_cfg = MODES[mode_name]
        proc = None

        print(f"\n{'='*70}", flush=True)
        print(f"  MODE: {mode_name}", flush=True)
        print(f"  {mode_cfg['description']}", flush=True)
        print(f"{'='*70}", flush=True)

        try:
            if not args.no_restart:
                print(f"  [setup] Killing existing server on :{args.port}...", flush=True)
                _kill_port(args.port)
                time.sleep(2)

                print(f"  [setup] Starting server ({mode_name})...", flush=True)
                proc = _start_server(args.model, args.port, mode_cfg["server_safekv_args"], mode_name)

                if not _wait_ready(args.port, timeout=420):
                    print(f"  [ERROR] Server not ready — skipping {mode_name}", flush=True)
                    if proc:
                        proc.terminate()
                    continue

                time.sleep(3)

            mode_result = run_mode(mode_name, mode_cfg, args.port, args.model, dataset, mw)
            all_results.append(mode_result)

        except KeyboardInterrupt:
            print("\n  [INTERRUPT]", flush=True)
            if proc:
                proc.terminate()
            break

        except Exception:
            print(f"\n  [ERROR] Mode {mode_name} failed:", flush=True)
            traceback.print_exc()

        finally:
            if not args.no_restart and proc is not None:
                print(f"\n  [teardown] Stopping server ({mode_name})...", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(3)

    if all_results:
        print_table(all_results)
        save_csv(all_results, csv_path)
        save_detail_csv(all_results, detail_path)
        plot_kv_timeline(all_results, png_path, args.model)
    else:
        print("\n  [WARN] No results collected.", flush=True)

    print(f"\n  Done.", flush=True)


if __name__ == "__main__":
    random.seed(42)
    main()
