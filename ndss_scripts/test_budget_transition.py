#!/usr/bin/env python3
"""
E2b: Budget Transition Time-Series Test
Demonstrates the SafeKV block lifecycle: Private → Shareable → Demoted → PP.

For each tenant-count N in {2, 5, 10, 20}:
  Phase 0 (Cold/Private): creator user inserts the shared context
  Phase 1 (Shareable):    N-1 other users each hit the shared context B times total
  Phase 2 (Demoted):      Budget exhausted → next cross-tenant hit gets cold TTFT
  Phase 3 (Re-insert/PP): creator_count reaches K → Permanently Public
  Phase 4 (Public):       all subsequent hits get shared TTFT

Metrics captured: TTFT per request, phase label, timestamp.
Summary: mean TTFT per phase + time_to_PP.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

SCRIPT_DIR = Path(__file__).parent.resolve()
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PYTHON = "/home/kec23008/.venv/bin/python3"

MODEL_CONFIGS = {
    "qwen32b": {
        "path": "/home/kec23008/Models/Qwen3-32B",
        "tp": 2, "dp": 1, "maxlen": 32768, "mem_frac": "0.85", "cuda_devices": "0,1",
    },
    "qwen30b": {
        "path": "/home/kec23008/Models/Qwen3-30B-A3B-Instruct-2507",
        "tp": 2, "dp": 1, "maxlen": 32768, "mem_frac": "0.90", "cuda_devices": "0,1",
    },
    "phi4": {
        "path": "/home/kec23008/Models/Phi-4",
        "tp": 1, "dp": 2, "maxlen": 16384, "mem_frac": "0.90", "cuda_devices": "0,1",
    },
}

COMPLETION_CONFIG = {"max_tokens": 64, "temperature": 0}
MODEL_COMPLETION_OVERRIDES: dict = {
    "qwen30b": {"enable_thinking": False},
    "qwen32b": {"enable_thinking": False},
    "phi4":    {"enable_thinking": False},
}

# Fixed non-PII shared context (same clinical document as throughput_ablation)
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

SHORT_QUESTION = "\n\nSummarize the key medication reconciliation steps in one sentence."


@dataclass
class ReqResult:
    req_id: int
    user_id: str
    phase: str
    ttft_ms: float
    total_ms: float
    timestamp: float
    error: Optional[str] = None


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


def _start_server(model: str, port: int, safekv_args: List[str]) -> subprocess.Popen:
    cfg = MODEL_CONFIGS[model]
    log_path = LOG_DIR / f"budget_transition_{model}_server.log"

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
    ] + safekv_args

    print(f"  [start] server cmd: {' '.join(cmd[:6])} ...", flush=True)
    print(f"  [log]   {log_path}", flush=True)
    log_fh = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env)


def _wait_ready(port: int, timeout: int = 480) -> bool:
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


def _send_one(port: int, model: str, req_id: int, user_id: str,
              prompt: str, phase: str) -> ReqResult:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    extra = MODEL_COMPLETION_OVERRIDES.get(model, {})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "user_id": user_id,
        **COMPLETION_CONFIG,
        **extra,
    }

    t0 = time.perf_counter()
    ttft: Optional[float] = None
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
                if content and ttft is None:
                    ttft = time.perf_counter() - t0
            except json.JSONDecodeError:
                pass
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - t0
    if ttft is None:
        ttft = elapsed

    return ReqResult(
        req_id=req_id,
        user_id=user_id,
        phase=phase,
        ttft_ms=ttft * 1000,
        total_ms=elapsed * 1000,
        timestamp=time.time(),
        error=error,
    )


def run_phase(port: int, model: str, user_ids: List[str],
              prompt: str, phase: str, n_requests: int,
              max_workers: int = 4, req_id_start: int = 0) -> List[ReqResult]:
    """Send n_requests round-robin across user_ids, return results in order."""
    results = []
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in range(n_requests):
            uid = user_ids[i % len(user_ids)]
            rid = req_id_start + i
            f = ex.submit(_send_one, port, model, rid, uid, prompt, phase)
            futures[f] = rid

        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as exc:
                results.append(ReqResult(
                    req_id=futures[f], user_id="?", phase=phase,
                    ttft_ms=float("nan"), total_ms=float("nan"),
                    timestamp=time.time(), error=str(exc)
                ))

    results.sort(key=lambda r: r.req_id)
    return results


def run_N(model: str, port: int, N: int, budget: int, K: int,
          detection_wait: float = 4.0) -> Tuple[List[ReqResult], Dict]:
    """
    Run the full block lifecycle for N tenants.
    Returns (all_results, summary_dict).
    """
    creator_id = f"tenant_creator"
    other_ids  = [f"tenant_{i}" for i in range(1, N)]  # N-1 other users

    shared_prompt = FIXED_SHARED_CONTEXT.strip() + SHORT_QUESTION

    all_results: List[ReqResult] = []
    req_ctr = 0
    t_start = time.time()

    # ── Phase 0: Cold/Private insert by creator ──────────────────────────────
    print(f"  [N={N}] Phase 0: Cold insert by creator ...", flush=True)
    res0 = _send_one(port, model, req_ctr, creator_id, shared_prompt, "cold")
    req_ctr += 1
    all_results.append(res0)
    ttft_cold = res0.ttft_ms
    print(f"    cold TTFT = {ttft_cold:.0f} ms", flush=True)

    # Wait for async detection to promote the shared context to Shareable
    print(f"  [N={N}] Waiting {detection_wait}s for async detection ...", flush=True)
    time.sleep(detection_wait)

    # ── Phase 1: Shareable — other users consume budget B ────────────────────
    # Send exactly `budget` requests round-robin across N-1 other users
    if N < 2:
        print(f"  [N={N}] Skipping Phase 1 (no other users)", flush=True)
        ttft_shareable = float("nan")
        time_shareable_start = time.time()
    else:
        print(f"  [N={N}] Phase 1: Shareable — {budget} requests across {len(other_ids)} users ...", flush=True)
        time_shareable_start = time.time()
        res1 = run_phase(
            port, model, other_ids, shared_prompt, "shareable",
            n_requests=budget, max_workers=min(N - 1, 8), req_id_start=req_ctr
        )
        req_ctr += len(res1)
        all_results.extend(res1)
        valid1 = [r.ttft_ms for r in res1 if r.error is None]
        ttft_shareable = sum(valid1) / len(valid1) if valid1 else float("nan")
        print(f"    shareable mean TTFT = {ttft_shareable:.0f} ms  ({len(valid1)} valid)", flush=True)

    # ── Phase 2: Demoted — budget exhausted, next hit is cold ────────────────
    print(f"  [N={N}] Phase 2: Demoted — single request after budget exhaustion ...", flush=True)
    uid_demoted = other_ids[0] if other_ids else creator_id
    res2 = _send_one(port, model, req_ctr, uid_demoted, shared_prompt, "demoted")
    req_ctr += 1
    all_results.append(res2)
    ttft_demoted = res2.ttft_ms
    print(f"    demoted TTFT = {ttft_demoted:.0f} ms", flush=True)

    # ── Phase 3: Re-insert by a second unique creator → creator_count=K → PP ─
    # creator_id already inserted once (Phase 0). We need creator_count >= K=2.
    # uid_demoted just re-inserted (Phase 2). If uid_demoted != creator_id,
    # that counts as creator 2 → PP.  Send one more to confirm PP.
    print(f"  [N={N}] Phase 3: PP verification — 5 requests ...", flush=True)
    t_pp_start = time.time()
    res3 = run_phase(
        port, model, [creator_id] + (other_ids[:1] if other_ids else []),
        shared_prompt, "pp",
        n_requests=5, max_workers=2, req_id_start=req_ctr
    )
    req_ctr += len(res3)
    all_results.extend(res3)
    valid3 = [r.ttft_ms for r in res3 if r.error is None]
    ttft_pp = sum(valid3) / len(valid3) if valid3 else float("nan")
    time_to_pp = t_pp_start - time_shareable_start
    print(f"    PP mean TTFT = {ttft_pp:.0f} ms  time_to_PP = {time_to_pp:.1f}s", flush=True)

    summary = {
        "N": N,
        "budget": budget,
        "K": K,
        "ttft_cold_ms": round(ttft_cold, 1),
        "ttft_shareable_ms": round(ttft_shareable, 1) if ttft_shareable == ttft_shareable else "nan",
        "ttft_demoted_ms": round(ttft_demoted, 1),
        "ttft_pp_ms": round(ttft_pp, 1) if ttft_pp == ttft_pp else "nan",
        "time_to_pp_s": round(time_to_pp, 1),
        "n_shareable_requests": budget,
        "total_requests": req_ctr,
    }
    return all_results, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi4", choices=list(MODEL_CONFIGS))
    ap.add_argument("--N-values", nargs="+", type=int, default=[2, 5, 10, 20],
                    help="Tenant counts to test")
    ap.add_argument("--budget", type=int, default=20,
                    help="SafeKV access budget B (cross-tenant hits before Demotion)")
    ap.add_argument("--K", type=int, default=2,
                    help="SafeKV creator threshold K (unique inserters to reach PP)")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--detection-wait", type=float, default=4.0,
                    help="Seconds to wait after cold insert for async detection")
    args = ap.parse_args()

    model = args.model
    budget = args.budget
    K = args.K
    port = args.port

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_path  = LOG_DIR / f"budget_transition_{model}_{timestamp}.csv"
    summary_path = LOG_DIR / f"budget_transition_summary_{model}_{timestamp}.csv"

    safekv_args = [
        "--safekv-access-budget", str(budget),
        "--safekv-creator-threshold", str(K),
    ]

    print(f"\n{'='*60}", flush=True)
    print(f"Budget Transition Test: model={model}  B={budget}  K={K}", flush=True)
    print(f"N-values: {args.N_values}", flush=True)
    print(f"{'='*60}\n", flush=True)

    _kill_port(port)
    proc = _start_server(model, port, safekv_args)

    if not _wait_ready(port):
        proc.terminate()
        sys.exit(1)

    # Give extra settling time
    time.sleep(5)

    all_detail: List[ReqResult] = []
    all_summary: List[Dict] = []

    for N in args.N_values:
        print(f"\n{'─'*50}", flush=True)
        print(f"N = {N} tenants", flush=True)
        try:
            results, summary = run_N(
                model, port, N, budget, K,
                detection_wait=args.detection_wait
            )
            all_detail.extend(results)
            all_summary.append(summary)
            print(f"  Summary: {summary}", flush=True)
        except Exception as exc:
            print(f"  [ERROR] N={N}: {exc}", flush=True)
            traceback.print_exc()

        # Brief pause between N-value runs
        time.sleep(10)

    # Write detail CSV
    with open(detail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "user_id", "phase", "ttft_ms", "total_ms",
                    "timestamp", "error"])
        for r in all_detail:
            w.writerow([r.req_id, r.user_id, r.phase,
                        f"{r.ttft_ms:.2f}", f"{r.total_ms:.2f}",
                        f"{r.timestamp:.3f}", r.error or ""])

    # Write summary CSV
    if all_summary:
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(all_summary[0].keys()))
            for row in all_summary:
                w.writerow(list(row.values()))

    # Console summary table
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS  model={model}  B={budget}  K={K}", flush=True)
    print(f"{'─'*60}", flush=True)
    print(f"{'N':>4}  {'Cold(ms)':>10}  {'Share(ms)':>10}  "
          f"{'Demote(ms)':>11}  {'PP(ms)':>8}  {'time_to_PP(s)':>14}", flush=True)
    for s in all_summary:
        print(f"{s['N']:>4}  {s['ttft_cold_ms']:>10}  "
              f"{str(s['ttft_shareable_ms']):>10}  "
              f"{s['ttft_demoted_ms']:>11}  "
              f"{str(s['ttft_pp_ms']):>8}  "
              f"{s['time_to_pp_s']:>14}", flush=True)
    print(f"{'─'*60}", flush=True)
    print(f"Detail CSV:  {detail_path}", flush=True)
    print(f"Summary CSV: {summary_path}", flush=True)

    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        pass
    _kill_port(port)


if __name__ == "__main__":
    main()
