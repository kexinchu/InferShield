#!/usr/bin/env python3
"""SafeKV Exp #7 – Revised component ablation (CPU-only invariant portion).

Experiment deliverable:
  "Revised component ablation"

Incrementally adds components to the unprotected baseline and measures:
  - invariant_violations (promotion integrity, namespace isolation)
  - cumulative_xt_hits   (cross-tenant hits after N eviction cycles)
  - first_xt_hit_cycle   (which cycle first cross-tenant hit occurs)

The 7-step ablation:
  1. SGLang (no SafeKV)           → unlimited cross-tenant sharing
  2. +private_namespace           → all nodes private, no cross-tenant hits
  3. +candidate_detection         → CANDIDATE state (simulated; Balanced mode)
  4. +balanced_in_node            → BUDGETED_SHARED with in-node counter only (no durable ledger)
  5. +balanced_durable_ledger     → BUDGETED_SHARED with DurableLedger (hits persist across eviction)
  6. +operator_auth_no_prewarm    → VERIFIED_PUBLIC via auth but no prewarm
  7. +full_registry_prewarm       → VERIFIED_PUBLIC + prewarm (full design)

TTFT and throughput are measured against a live server in a separate run.
This script covers only the deterministic invariant portion (CPU-only).

Usage:
  python exp7_revised_ablation.py \
      --n-cycles 20 --budget-B 10 \
      --output results/exp7/ablation_invariants.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Stub memory_pool ──────────────────────────────────────────────────────────
_pool_stub = types.ModuleType("sglang.srt.mem_cache.memory_pool")
_pool_stub.ReqToTokenPool = object  # type: ignore[attr-defined]
_pool_stub.TokenToKVPoolAllocator = object  # type: ignore[attr-defined]
sys.modules["sglang.srt.mem_cache.memory_pool"] = _pool_stub

import torch  # noqa: E402

from sglang.srt.mem_cache.radix_cache import RadixCache  # noqa: E402
from sglang.srt.mem_cache.safekv_policy import (  # noqa: E402
    DurableLedger,
    PublicRegistry,
    Visibility,
)

BUDGET_B = 10
OPERATOR_KEY = b"safekv-exp7-key"
MODEL_ID = "phi-4-14b"
TOKENIZER = "phi-4-14b-tokenizer"
TOKENS = list(range(100, 164))

CSV_FIELDS = (
    "ablation_step", "step_name",
    "n_cycles", "budget_B",
    "total_xt_hits", "first_xt_hit_cycle",
    "max_xt_hits_per_cycle", "budgeted_respected",
    "strict_invariant",
    "unauthorized_private_hits", "budgeted_hits", "authorized_public_hits",
    "budget_scope", "public_ready_before_workload",
    "materialization_requests",
    "public_created", "public_aliases_victim",
    "verdict",
)


def _make_cache(mode: str, ledger_path: Optional[str] = None) -> RadixCache:
    from unittest.mock import MagicMock
    alloc = MagicMock()
    alloc.device = torch.device("cpu")
    judge = MagicMock()
    return RadixCache(
        req_to_token_pool=None,
        token_to_kv_pool_allocator=alloc,
        page_size=1,
        private_judge_client=judge,
        disable=False,
        access_budget_B=BUDGET_B,
        safekv_mode=mode,
        operator_key=OPERATOR_KEY.decode(),
        policy_epoch=1,
        model_id=MODEL_ID,
        tokenizer_version=TOKENIZER,
        ledger_path=ledger_path,
    )


def _insert(cache: RadixCache, user_id: str, visibility: str = "private") -> None:
    val = torch.tensor(list(range(len(TOKENS))), dtype=torch.int64)
    cache.insert(key=TOKENS, value=val, user_id=user_id)
    if visibility in ("budgeted_shared", "candidate"):
        root = cache._private_root(user_id, create=False)
        if root:
            for node in list(root.children.values()):
                if visibility == "candidate":
                    # Candidate is still private: detection has not yet authorized
                    # any cross-tenant reuse.
                    node.visibility = Visibility.CANDIDATE.value
                else:
                    fp = cache._node_fingerprint(TOKENS)
                    charged = cache.ledger.charged_hits(fp)
                    node.access_budget = charged
                    if charged >= cache.access_budget_B:
                        node.visibility = Visibility.EXHAUSTED_PRIVATE.value
                        node.permanently_private = True
                    else:
                        node.visibility = Visibility.BUDGETED_SHARED.value


def _hit(cache: RadixCache, user_id: str) -> Tuple[bool, str]:
    """Return (hit, namespace) where namespace is 'private', 'public', or 'none'."""
    result, _ = cache.match_prefix(key=TOKENS, user_id=user_id)
    if len(result) == 0:
        return False, "none"
    # Inspect events to determine served namespace.
    snap = cache.metrics.snapshot()
    events = list(snap["events"])
    for ev in reversed(events):
        if ev.name == "lookup" and ev.attributes.get("requester") == user_id:
            ns = ev.attributes.get("served_namespace", "unknown")
            return True, ns
    return True, "unknown"


def _evict(cache: RadixCache) -> None:
    for root in list(cache.private_roots.values()):
        for node in list(root.children.values()):
            node.value = None
    cache.private_roots.clear()
    cache.evictable_size_ = 0


def run_ablation_step(
    step_name: str,
    n_cycles: int,
    ledger_path: Optional[str] = None,
) -> Dict:
    """Run one ablation configuration and return metrics."""
    # Map step_name → cache mode and visibility promotion.
    if step_name == "sglan":
        # No SafeKV at all: all nodes are shareable (simulated by "balanced" with
        # no budget, but disable private namespace by sharing one user_id).
        cache = _make_cache(mode="balanced", ledger_path=ledger_path)
        owner, attacker = "shared-user", "shared-user"  # same user = no isolation
    elif step_name in ("private_namespace", "candidate_detection"):
        cache = _make_cache(mode="strict", ledger_path=ledger_path)
        owner, attacker = "victim", "attacker"
    elif step_name in ("balanced_in_node", "balanced_durable_ledger"):
        cache = _make_cache(mode="balanced", ledger_path=ledger_path)
        owner, attacker = "victim", "attacker"
    elif step_name in ("operator_auth_no_prewarm", "full_registry_prewarm"):
        cache = _make_cache(mode="strict", ledger_path=ledger_path)
        owner, attacker = "victim", "attacker"
    else:
        raise ValueError(f"Unknown step: {step_name}")

    total_xt_hits = 0
    total_private_hits = 0
    first_xt_hit_cycle = None
    max_hits_per_cycle = 0
    public_created = 0
    public_aliases_victim = 0
    public_ready_before_workload = 0
    materialization_requests = 0

    # Authorization and prewarming are control-plane actions and happen once
    # per object, not once per eviction cycle.
    public_auth = None
    if step_name in ("operator_auth_no_prewarm", "full_registry_prewarm"):
        reg = PublicRegistry(OPERATOR_KEY, policy_epoch=1)
        import time as _time
        public_auth = reg.issue(
            "pub-obj", "ctrl", MODEL_ID, TOKENIZER, TOKENS[:-1],
            _time.time() + 7200,
        )
        if step_name == "full_registry_prewarm":
            cache.insert(
                key=TOKENS[:-1],
                value=torch.tensor(list(range(len(TOKENS) - 1)), dtype=torch.int64),
                user_id="operator-prewarm",
                authorization=public_auth.to_dict(),
            )
            public_created = int(bool(cache.public_roots))
            public_ready_before_workload = public_created

    for cycle in range(n_cycles):
        # Insert.
        if step_name == "sglan":
            _insert(cache, owner, "private")
        elif step_name == "private_namespace":
            _insert(cache, owner, "private")
        elif step_name == "candidate_detection":
            _insert(cache, owner, "candidate")  # same as budgeted for this test
        elif step_name == "balanced_in_node":
            _insert(cache, owner, "budgeted_shared")
        elif step_name == "balanced_durable_ledger":
            _insert(cache, owner, "budgeted_shared")  # restores from ledger via _insert_budgeted logic
        elif step_name in ("operator_auth_no_prewarm", "full_registry_prewarm"):
            # The victim's private object is always independent of Public KV.
            _insert(cache, owner, "private")
            if step_name == "operator_auth_no_prewarm" and public_created == 0:
                # Reactive materialization: authorization exists, but the Public
                # object appears only when the first authorized request arrives.
                assert public_auth is not None
                cache.insert(
                    key=TOKENS[:-1],
                    value=torch.tensor(list(range(len(TOKENS) - 1)), dtype=torch.int64),
                    user_id=owner,
                    authorization=public_auth.to_dict(),
                )
                public_created = int(bool(cache.public_roots))
                materialization_requests += 1

            # Compare actual KV tensor storage, not token-key equality: Public
            # and Private may cover equal tokens but must never alias storage.
            victim_root = cache._private_root(owner, create=False)
            pub_roots = list(cache.public_roots.values())
            if victim_root and pub_roots:
                victim_ptrs = {
                    n.value.data_ptr()
                    for n in victim_root.children.values()
                    if isinstance(n.value, torch.Tensor)
                }
                public_ptrs = {
                    n.value.data_ptr()
                    for pr in pub_roots
                    for n in pr.children.values()
                    if isinstance(n.value, torch.Tensor)
                }
                public_aliases_victim = int(bool(victim_ptrs & public_ptrs))

        # Measure cross-tenant hits; distinguish public vs private namespace.
        hits_this_cycle = 0
        private_hits_this_cycle = 0
        for _ in range(BUDGET_B + 2):
            hit, ns = _hit(cache, attacker)
            if hit:
                hits_this_cycle += 1
                if ns in ("private", "unknown"):
                    private_hits_this_cycle += 1
        if hits_this_cycle > 0 and first_xt_hit_cycle is None:
            first_xt_hit_cycle = cycle
        total_xt_hits += hits_this_cycle
        total_private_hits += private_hits_this_cycle
        max_hits_per_cycle = max(max_hits_per_cycle, hits_this_cycle)

        cache.ledger.flush()
        _evict(cache)
        if step_name == "balanced_in_node":
            # Volatile in-node accounting is residency-scoped.  Eviction loses
            # the charge, so the next insertion receives a fresh budget.
            cache.ledger.reset()

    if step_name == "sglan":
        unauthorized_private_hits = total_xt_hits
        budgeted_hits = 0
        authorized_public_hits = 0
        budget_scope = "none"
    elif step_name in ("private_namespace", "candidate_detection"):
        unauthorized_private_hits = total_xt_hits
        budgeted_hits = 0
        authorized_public_hits = 0
        budget_scope = "private"
    elif step_name in ("balanced_in_node", "balanced_durable_ledger"):
        unauthorized_private_hits = 0
        budgeted_hits = total_xt_hits
        authorized_public_hits = 0
        budget_scope = (
            "per_residency"
            if step_name == "balanced_in_node"
            else "per_accounting_epoch"
        )
    else:
        unauthorized_private_hits = total_private_hits
        budgeted_hits = 0
        authorized_public_hits = total_xt_hits - total_private_hits
        budget_scope = "authorized_public"

    private_namespace_invariant = (
        None if step_name == "sglan" else unauthorized_private_hits == 0
    )

    # Check the bound in the scope actually implemented by each variant.
    if step_name == "balanced_in_node":
        budgeted_respected = max_hits_per_cycle <= BUDGET_B
    elif step_name == "balanced_durable_ledger":
        budgeted_respected = total_xt_hits <= BUDGET_B
    else:
        budgeted_respected = None

    verdict = "pass"
    if step_name == "sglan":
        verdict = "expected_violations"
    elif private_namespace_invariant is False:
        verdict = "fail"
    elif budgeted_respected is False:
        verdict = "fail"

    return {
        "step_name": step_name,
        "n_cycles": n_cycles,
        "budget_B": BUDGET_B,
        "total_xt_hits": total_xt_hits,
        "first_xt_hit_cycle": first_xt_hit_cycle if first_xt_hit_cycle is not None else -1,
        "max_xt_hits_per_cycle": max_hits_per_cycle,
        "budgeted_respected": budgeted_respected,
        "strict_invariant": private_namespace_invariant,
        "unauthorized_private_hits": unauthorized_private_hits,
        "budgeted_hits": budgeted_hits,
        "authorized_public_hits": authorized_public_hits,
        "budget_scope": budget_scope,
        "public_ready_before_workload": public_ready_before_workload,
        "materialization_requests": materialization_requests,
        "public_created": public_created,
        "public_aliases_victim": public_aliases_victim,
        "verdict": verdict,
    }


STEPS = [
    ("1", "sglan"),
    ("2", "private_namespace"),
    ("3", "candidate_detection"),
    ("4", "balanced_in_node"),
    ("5", "balanced_durable_ledger"),
    ("6", "operator_auth_no_prewarm"),
    ("7", "full_registry_prewarm"),
]


def main() -> None:
    global BUDGET_B
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--budget-B", type=int, default=BUDGET_B)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "exp7_ablation_invariants.csv",
    )
    args = parser.parse_args()
    BUDGET_B = args.budget_B
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" SafeKV Exp #7 – Revised Component Ablation")
    print(f" budget B={BUDGET_B}  n_cycles={args.n_cycles}")
    print(f"{'='*60}\n")

    rows = []
    for step_num, step_name in STEPS:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            ledger_path = tf.name
        # DurableLedger distinguishes a missing store (first initialization)
        # from an existing but empty/corrupt JSON file (fail closed).
        os.unlink(ledger_path)
        try:
            metrics = run_ablation_step(step_name, args.n_cycles, ledger_path)
        finally:
            try:
                os.unlink(ledger_path)
            except Exception:
                pass

        row = {"ablation_step": step_num, **metrics}
        rows.append(row)

        vi = metrics["strict_invariant"]
        br = metrics["budgeted_respected"]
        print(
            f"  Step {step_num}: {step_name:<35}"
            f"private={metrics['unauthorized_private_hits']:<4} "
            f"budgeted={metrics['budgeted_hits']:<4} "
            f"public={metrics['authorized_public_hits']:<4} "
            f"strict_inv={vi}  budget_ok={br}  verdict={metrics['verdict']}"
        )

    # Write CSV.
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults → {args.output}")

    # LaTeX table rows.
    print("\n=== LaTeX table rows ===")
    for row in rows:
        print(
            f"  {row['ablation_step']} & {row['step_name']}"
            f" & {row['total_xt_hits']}"
            f" & {row['strict_invariant'] if row['strict_invariant'] is not None else '-'}"
            f" & {row['budgeted_respected'] if row['budgeted_respected'] is not None else '-'}"
            f" & {row['public_created']} \\\\"
        )

    all_pass = all(r["verdict"] in ("pass", "expected_violations") for r in rows)
    print(f"\nOverall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
