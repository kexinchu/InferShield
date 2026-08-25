#!/usr/bin/env python3
"""SafeKV Exp #8 (appendix) – authorization validation and revocation matrix.

Tests 11 authorization conditions against a live server and also runs a
CPU-only revocation-propagation latency microbenchmark.

Experiment deliverable: appendix table "Authorization validation and revocation
matrix" with rows for each condition and columns for:
  - public_lookup_result:   allowed | denied
  - deny_reason:            OK | INVALID_MAC | WRONG_MODEL | ...
  - private_fallback:       0/1 (did requester get private cache hit?)
  - victim_relabels:        must be 0 for all conditions
  - revocation_latency_us:  CPU-only, for active_revoked row only

All invalid conditions must fail closed without relabeling victim private
state.  The valid condition must create a separate Public object.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import hmac as _hmac
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from transformers import AutoTokenizer

from sglang.srt.mem_cache.safekv_policy import (
    PublicAuthorization,
    PublicRegistry,
    VerificationResult,
)

# ── Conditions ────────────────────────────────────────────────────────────────
#  Maps P8 condition names → (public_lookup_expected, note)
P8_CONDITIONS: Dict[str, Tuple[str, str]] = {
    "absent": ("denied", "no auth object supplied"),
    "forged_mac": ("denied", "HMAC signed with wrong key"),
    "malformed": ("denied", "MAC field corrupted after issuance"),
    "wrong_model": ("denied", "model_id mismatch in otherwise valid auth"),
    "wrong_tokenizer": ("denied", "tokenizer_version mismatch"),
    "tampered_length": ("denied", "signed authorization length field modified"),
    "wrong_tokens_same_length": ("denied", "same-length token fingerprint mismatch"),
    "expired": ("denied", "expires_at is in the past with valid MAC"),
    "stale_epoch": ("denied", "policy_epoch != current epoch"),
    "revoked": ("denied", "auth issued with revoked=True flag"),
    "valid": ("allowed", "valid operator authorization (positive control)"),
    "valid_shorter_prefix": (
        "allowed",
        "valid authorization for a shorter prefix of the submitted request",
    ),
    "active_revoked": ("denied", "CPU-only: install then revoke, verify denial"),
}

CSV_FIELDS = (
    "model",
    "trial_id",
    "condition",
    "public_lookup_result",
    "deny_reason",
    "public_created",
    "victim_relabels",
    "private_fallback",
    "victim_private_still_owned",
    "post_revoke_public_hit",
    "stale_reinstall_denied",
    "verdict",
    "error",
)


# ── Auth builders ─────────────────────────────────────────────────────────────

def _build_valid_auth(
    tokens: List[int],
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    object_id: str,
    epoch: int = 1,
    revoked: bool = False,
    key_override: Optional[bytes] = None,
    future: float = 3600.0,
) -> PublicAuthorization:
    key = key_override if key_override is not None else operator_key.encode("utf-8")
    registry = PublicRegistry(key, policy_epoch=epoch)
    return registry.issue(
        public_object_id=object_id,
        issuer="safekv-exp8-ctrl",
        model_id=model_id,
        tokenizer_version=tokenizer_version,
        token_ids=tokens,
        expires_at=time.time() + future,
        revoked=revoked,
    )


def _build_expired_auth(
    tokens: List[int],
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    object_id: str,
    epoch: int = 1,
) -> PublicAuthorization:
    """Issue an auth with a valid MAC but an already-elapsed expires_at."""
    key = operator_key.encode("utf-8")
    past_time = time.time() - 3600.0  # 1 hour ago

    payload, token_tuple = PublicRegistry._prefix_payload(
        model_id, tokenizer_version, tokens
    )
    fingerprint = hashlib.sha256(payload).hexdigest()
    auth_payload = PublicRegistry._authorization_payload(
        object_id, "safekv-exp8-ctrl", payload, epoch, past_time, False
    )
    mac = _hmac.new(key, auth_payload, hashlib.sha256).hexdigest()

    return PublicAuthorization(
        public_object_id=object_id,
        issuer="safekv-exp8-ctrl",
        model_id=model_id,
        tokenizer_version=tokenizer_version,
        prefix_token_length=len(token_tuple),
        prefix_fingerprint=fingerprint,
        policy_epoch=epoch,
        expires_at=past_time,
        revoked=False,
        mac=mac,
    )


def make_auth_p8(
    condition: str,
    tokens: List[int],
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    object_id: str,
) -> Optional[PublicAuthorization]:
    """Return the PublicAuthorization for the given P8 condition (or None for absent)."""
    if condition == "absent":
        return None

    if condition == "forged_mac":
        return _build_valid_auth(
            tokens, model_id, tokenizer_version, operator_key, object_id,
            key_override=b"attacker-forged-key",
        )

    if condition == "malformed":
        # Issue a legitimately-signed auth then corrupt the MAC field.
        auth = _build_valid_auth(tokens, model_id, tokenizer_version, operator_key, object_id)
        return dataclasses.replace(auth, mac="de" * 32)

    if condition == "wrong_model":
        return _build_valid_auth(
            tokens, "wrong-model-x/unknown", tokenizer_version, operator_key, object_id
        )

    if condition == "wrong_tokenizer":
        return _build_valid_auth(
            tokens, model_id, "wrong-tokenizer-v99", operator_key, object_id
        )

    if condition == "tampered_length":
        auth = _build_valid_auth(
            tokens, model_id, tokenizer_version, operator_key, object_id
        )
        return dataclasses.replace(
            auth, prefix_token_length=auth.prefix_token_length + 1
        )

    if condition == "wrong_tokens_same_length":
        wrong_tokens = list(tokens)
        wrong_tokens[-1] = wrong_tokens[-1] + 1
        return _build_valid_auth(
            wrong_tokens, model_id, tokenizer_version, operator_key, object_id
        )

    if condition == "expired":
        return _build_expired_auth(tokens, model_id, tokenizer_version, operator_key, object_id)

    if condition == "stale_epoch":
        return _build_valid_auth(
            tokens, model_id, tokenizer_version, operator_key, object_id, epoch=0
        )

    if condition == "revoked":
        return _build_valid_auth(
            tokens, model_id, tokenizer_version, operator_key, object_id, revoked=True
        )

    if condition == "valid":
        return _build_valid_auth(tokens, model_id, tokenizer_version, operator_key, object_id)

    if condition == "valid_shorter_prefix":
        shorter = tokens[:-3] if len(tokens) > 3 else tokens[:1]
        return _build_valid_auth(
            shorter, model_id, tokenizer_version, operator_key, object_id
        )

    if condition == "active_revoked":
        # The active_revoked test is handled as a CPU-only offline test.
        # Return a valid auth so the live test also runs.
        return _build_valid_auth(tokens, model_id, tokenizer_version, operator_key, object_id)

    raise ValueError(f"unknown condition: {condition}")


# ── CPU-only revocation latency microbenchmark ────────────────────────────────

def revocation_latency_benchmark(
    tokens: List[int],
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    object_id: str,
    n_samples: int = 200,
) -> Dict[str, float]:
    """Measure how quickly PublicRegistry.revoke() becomes effective.

    Because the registry is in-process and the lock is held atomically, the
    revoke should be reflected on the very next verify() call.  We measure
    the wall-clock time from revoke() return to the first denial.
    """
    key = operator_key.encode("utf-8")
    results = []
    for _ in range(n_samples):
        registry = PublicRegistry(key, policy_epoch=1)
        auth = registry.issue(
            public_object_id=object_id,
            issuer="safekv-exp8-ctrl",
            model_id=model_id,
            tokenizer_version=tokenizer_version,
            token_ids=tokens,
            expires_at=time.time() + 3600,
        )
        # Verify it is accepted.
        ok = registry.verify(auth, model_id, tokenizer_version, tokens)
        assert ok.valid, f"pre-revoke verify should pass: {ok.reason}"

        t0 = time.perf_counter()
        registry.revoke(object_id)
        result = registry.verify(auth, model_id, tokenizer_version, tokens)
        t1 = time.perf_counter()

        assert not result.valid, f"post-revoke verify should fail: {result.reason}"
        results.append((t1 - t0) * 1e6)  # µs

    results.sort()
    n = len(results)
    return {
        "mean_us": sum(results) / n,
        "p50_us": results[n // 2],
        "p95_us": results[int(n * 0.95)],
        "p99_us": results[int(n * 0.99)],
        "max_us": results[-1],
        "n": n,
    }


def revocation_linearization_benchmark(
    tokens: List[int],
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    n_samples: int = 500,
) -> Dict[str, int]:
    """Race verify against revoke and check the post-revoke boundary.

    The concurrent verify may linearize before revoke (allowed) or after it
    (denied).  Every verification started after both threads complete must deny.
    """
    key = operator_key.encode("utf-8")
    before_allowed = 0
    after_denied = 0
    post_revoke_denied = 0

    for i in range(n_samples):
        object_id = f"race-object-{i}"
        registry = PublicRegistry(key, policy_epoch=1)
        auth = registry.issue(
            public_object_id=object_id,
            issuer="safekv-exp8-ctrl",
            model_id=model_id,
            tokenizer_version=tokenizer_version,
            token_ids=tokens,
            expires_at=time.time() + 3600,
        )
        barrier = threading.Barrier(2)
        result: List[VerificationResult] = []

        def do_verify() -> None:
            barrier.wait()
            result.append(
                registry.verify(auth, model_id, tokenizer_version, tokens)
            )

        thread = threading.Thread(target=do_verify)
        thread.start()
        barrier.wait()
        registry.revoke(object_id)
        thread.join()

        if result[0].valid:
            before_allowed += 1
        else:
            after_denied += 1
        if not registry.verify(auth, model_id, tokenizer_version, tokens).valid:
            post_revoke_denied += 1

    return {
        "n": n_samples,
        "concurrent_verify_allowed_before_revoke": before_allowed,
        "concurrent_verify_denied_after_revoke": after_denied,
        "post_revoke_verify_denied": post_revoke_denied,
    }


# ── Server helpers ────────────────────────────────────────────────────────────

class ExperimentClient:
    def __init__(self, server: str, timeout: float = 120.0):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def flush(self) -> None:
        response = requests.post(f"{self.server}/flush_cache", timeout=self.timeout)
        if response.status_code not in (200, 400):
            response.raise_for_status()
        if response.status_code == 400:
            # Some builds report an already-empty cache as 400.  Accept that
            # response only after confirming there is no residual variant.
            if not self.snapshot()["variants"]:
                return
            response.raise_for_status()
        for _ in range(100):
            snap = self.snapshot()
            if not snap["variants"]:
                return
            time.sleep(0.05)
        raise RuntimeError("cache did not empty after flush")

    def snapshot(self) -> Dict[str, object]:
        response = requests.get(f"{self.server}/get_server_info", timeout=self.timeout)
        response.raise_for_status()
        states = response.json()["internal_states"]
        if len(states) != 1:
            raise RuntimeError("requires dp_size=1")
        return states[0]["safekv"]

    def generate(
        self,
        token_ids: List[int],
        principal: str,
        authorization: Optional[PublicAuthorization] = None,
    ) -> Dict[str, object]:
        params: Dict[str, object] = {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "user_id": principal,
        }
        if authorization is not None:
            params["safekv_public_authorization"] = authorization.to_dict()
        response = requests.post(
            f"{self.server}/generate",
            json={"input_ids": token_ids, "sampling_params": params, "stream": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def _identities(snapshot, visibility: str, identity: Optional[str] = None) -> list:
    variants = [v for v in snapshot["variants"] if v["namespace_visibility"] == visibility]
    if identity is not None:
        variants = [v for v in variants if v["namespace_identity"] == identity]
    return variants


# ── Trial evaluation ──────────────────────────────────────────────────────────

def evaluate_p8_trial(
    client: ExperimentClient,
    model: str,
    token_ids: List[int],
    trial_id: int,
    condition: str,
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
) -> Dict[str, object]:
    victim = f"p8-victim-{trial_id}"
    attacker = f"p8-attacker-{trial_id}"
    stale_attacker = f"p8-stale-attacker-{trial_id}"
    object_id = f"p8-pub-{trial_id}-{condition}"
    auth_tokens = token_ids[:-1]

    auth = make_auth_p8(condition, auth_tokens, model_id, tokenizer_version, operator_key, object_id)
    client.flush()

    # Victim inserts first.
    client.generate(token_ids, victim)

    # Attacker submits with the specified auth condition.
    # For active_revoked: we use a valid auth for the first request (to install),
    # then we reset the epoch by re-issuing a revoked version for the check.
    check_auth = auth
    post_revoke_probe = f"p8-post-revoke-probe-{trial_id}"
    if condition == "active_revoked":
        # First prewarm with valid auth so the server installs it.
        client.generate(token_ids, attacker, auth)
        # Now submit a revoked version of the same auth.
        revoked_auth = _build_valid_auth(
            auth_tokens, model_id, tokenizer_version, operator_key, object_id, revoked=True
        )
        check_auth = revoked_auth

    client.generate(token_ids, attacker, check_auth)
    if condition == "active_revoked":
        # Revocation must hide an already materialized Public object even from
        # a request with no authorization, and the old valid manifest must not
        # reinstall it.
        client.generate(token_ids, post_revoke_probe)
        client.generate(token_ids, stale_attacker, auth)

    snap = client.snapshot()
    counters = snap["counters"]
    events = snap["events"]

    victim_variants = _identities(snap, "private", victim)
    public_variants = _identities(snap, "verified_public")

    # Derive outcomes.
    public_created = int(counters["public_object_created"])
    victim_relabels = sum(
        1 for v in snap["variants"]
        if v.get("creator_id") == victim and v["namespace_identity"] != victim
    )

    # Auth verify events for the attacker's last request.
    auth_events = [
        e for e in events
        if e["name"] == "authorization_verified"
    ]
    # The most recent auth_verify event gives the deny_reason.
    deny_reason = "none"
    if auth_events:
        last = auth_events[-1]["attributes"]
        deny_reason = last.get("reason", "none")
        accepted = bool(last.get("accepted", False))
    else:
        accepted = False

    public_lookup_result = "allowed" if accepted else "denied"

    # Private fallback: attacker got a private cache hit (requester == owner).
    private_probe_principal = (
        stale_attacker if condition == "active_revoked" else attacker
    )
    attacker_private_hit = any(
        e["name"] == "lookup"
        and e["attributes"].get("requester") == private_probe_principal
        and e["attributes"].get("served_namespace") == "private"
        and e["attributes"].get("hit")
        for e in events
    )
    private_fallback = int(attacker_private_hit)
    victim_owned = int(bool(victim_variants))
    post_revoke_public_hit = int(
        any(
            e["name"] == "lookup"
            and e["attributes"].get("requester") == post_revoke_probe
            and e["attributes"].get("served_namespace") == "public"
            and e["attributes"].get("hit")
            for e in events
        )
    )
    stale_reinstall_denied = int(
        condition != "active_revoked"
        or (
            public_lookup_result == "denied"
            and deny_reason == "revoked"
        )
    )

    # Verdict logic.
    # "valid" positive control: last request must be allowed and create a Public object.
    # "active_revoked": the first request (valid) creates a Public object; the CHECK
    #   request (revoked) must be denied.  So public_created may be ≥1 from the
    #   first prewarm — what matters is the final lookup was denied.
    expected_allow = condition in ("valid", "valid_shorter_prefix")
    if expected_allow:
        verdict = "pass" if (
            public_lookup_result == "allowed"
            and public_created >= 1
            and victim_relabels == 0
        ) else "fail"
    elif condition == "active_revoked":
        # Only the CHECK request result matters; Public may already exist from prewarm.
        verdict = "pass" if (
            public_lookup_result == "denied"
            and victim_relabels == 0
            and post_revoke_public_hit == 0
            and stale_reinstall_denied == 1
        ) else "fail"
    else:
        verdict = "pass" if (
            public_lookup_result == "denied"
            and public_created == 0
            and victim_relabels == 0
        ) else "fail"

    return {
        "model": model,
        "trial_id": trial_id,
        "condition": condition,
        "public_lookup_result": public_lookup_result,
        "deny_reason": deny_reason,
        "public_created": public_created,
        "victim_relabels": victim_relabels,
        "private_fallback": private_fallback,
        "victim_private_still_owned": victim_owned,
        "post_revoke_public_hit": post_revoke_public_hit,
        "stale_reinstall_denied": stale_reinstall_denied,
        "verdict": verdict,
        "error": "",
    }


# ── Prefix loading ────────────────────────────────────────────────────────────

def load_prefixes(dataset: Path, tokenizer, count: int, seed: int) -> List[List[int]]:
    import random
    rng = random.Random(seed)
    texts = []
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = item.get("source_text") or item.get("text")
            if text and len(text.strip()) >= 40:
                texts.append(text.strip())
    rng.shuffle(texts)
    prefixes = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)[:128]
        if len(ids) >= 16:
            prefixes.append(ids)
        if len(prefixes) >= count:
            break
    if len(prefixes) < count:
        raise RuntimeError(f"need {count} prefixes, found {len(prefixes)}")
    return prefixes


def load_completed(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["condition"], int(row["trial_id"]))
            for row in csv.DictReader(handle)
            if not row.get("error")
        }


def append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="http://host:port")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parents[1] / "datasets" / "english_pii_43k.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operator-key",
        default=os.environ.get("SAFEKV_OPERATOR_KEY", "safekv-exp8-operator-key"),
    )
    parser.add_argument(
        "--revocation-bench-only",
        action="store_true",
        help="Only run the CPU-only revocation latency benchmark",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # CPU-only revocation latency benchmark — always runs.
    print("\n=== Revocation propagation latency (CPU-only) ===")
    # Use a small synthetic token sequence for the benchmark.
    bench_tokens = list(range(32, 96))
    bench_stats = revocation_latency_benchmark(
        bench_tokens,
        model_id=args.model,
        tokenizer_version=args.model_path,
        operator_key=args.operator_key,
        object_id="bench-object",
        n_samples=500,
    )
    bench_stats["linearization"] = revocation_linearization_benchmark(
        bench_tokens,
        model_id=args.model,
        tokenizer_version=args.model_path,
        operator_key=args.operator_key,
        n_samples=500,
    )
    print(
        f"  mean={bench_stats['mean_us']:.2f}µs  "
        f"p50={bench_stats['p50_us']:.2f}µs  "
        f"p95={bench_stats['p95_us']:.2f}µs  "
        f"p99={bench_stats['p99_us']:.2f}µs  "
        f"max={bench_stats['max_us']:.2f}µs  "
        f"n={bench_stats['n']}"
    )
    bench_out = args.output.parent / (args.output.stem + "_revocation_bench.json")
    bench_out.parent.mkdir(parents=True, exist_ok=True)
    bench_out.write_text(json.dumps(bench_stats, indent=2))
    print(f"  Saved → {bench_out}")

    if args.revocation_bench_only:
        return

    # Live server tests.
    client = ExperimentClient(args.server)
    server_info = requests.get(
        f"{args.server.rstrip('/')}/get_model_info", timeout=30
    ).json()
    tokenizer_version = server_info["tokenizer_path"]

    prefixes = load_prefixes(args.dataset, tokenizer, args.trials, args.seed)
    completed = load_completed(args.output)

    conditions = list(P8_CONDITIONS.keys())
    total = len(conditions) * args.trials
    done = len(completed)

    print(f"\n=== Server tests: {total} trials ({len(conditions)} conditions × {args.trials}) ===")

    for condition in conditions:
        for trial_id in range(args.trials):
            if (condition, trial_id) in completed:
                done += 1
                continue
            try:
                row = evaluate_p8_trial(
                    client,
                    args.model,
                    prefixes[trial_id],
                    trial_id,
                    condition,
                    args.model,
                    tokenizer_version,
                    args.operator_key,
                )
            except Exception as exc:
                row = {field: "" for field in CSV_FIELDS}
                row.update({
                    "model": args.model,
                    "trial_id": trial_id,
                    "condition": condition,
                    "verdict": "error",
                    "error": repr(exc),
                })
            append_row(args.output, row)
            done += 1
            print(
                f"[{done}/{total}] condition={condition} trial={trial_id} "
                f"result={row.get('public_lookup_result','')} "
                f"reason={row.get('deny_reason','')} "
                f"verdict={row.get('verdict','')}",
                flush=True,
            )
            if row.get("error") and row.get("verdict") == "error":
                raise RuntimeError(row["error"])

    # Print summary table.
    print("\n=== Summary (condition → result across all trials) ===")
    if args.output.exists():
        from collections import Counter
        summary: Dict[str, Counter] = {}
        with args.output.open(newline="") as handle:
            for row in csv.DictReader(handle):
                cond = row["condition"]
                if cond not in summary:
                    summary[cond] = Counter()
                summary[cond][row["verdict"]] += 1
        for cond, counts in summary.items():
            expected = P8_CONDITIONS.get(cond, ("?", ""))[0]
            print(
                f"  {cond:<20} expected={expected:<7} "
                f"pass={counts.get('pass', 0)}  "
                f"fail={counts.get('fail', 0)}  "
                f"error={counts.get('error', 0)}"
            )


if __name__ == "__main__":
    main()
