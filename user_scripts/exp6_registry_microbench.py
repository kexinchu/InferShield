#!/usr/bin/env python3
"""SafeKV Exp #6 – Registry and ledger microbenchmarks (CPU-only).

Experiment deliverable:
  "Registry and ledger microbenchmarks"

Measures:
  - Public Registry: verify, fingerprint lookup, install, revoke, advance epoch
  - DurableLedger:   add_hits (in-memory), flush (fsync), charged_hits, reset
  - Keyed fingerprint computation: PublicRegistry.fingerprint()
  - Authorization MAC computation: PublicRegistry.issue()
  - Registry lookup throughput vs registry size (1, 10, 100, 1000 entries)

All timings in µs.  CPU metadata sizes reported in bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Stub memory_pool so radix_cache can be imported ──────────────────────────
_pool_stub = types.ModuleType("sglang.srt.mem_cache.memory_pool")
_pool_stub.ReqToTokenPool = object  # type: ignore[attr-defined]
_pool_stub.TokenToKVPoolAllocator = object  # type: ignore[attr-defined]
sys.modules["sglang.srt.mem_cache.memory_pool"] = _pool_stub

from sglang.srt.mem_cache.safekv_policy import (  # noqa: E402
    DurableLedger,
    PublicAuthorization,
    PublicRegistry,
)

OPERATOR_KEY = b"safekv-exp6-bench-key"
MODEL_ID = "phi-4-14b"
TOKENIZER = "phi-4-14b-tokenizer"
POLICY_EPOCH = 1


# ── Benchmark harness ─────────────────────────────────────────────────────────

def _bench(fn, n: int = 1000) -> Dict[str, float]:
    """Run fn() n times and return timing statistics in µs."""
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)
    samples.sort()
    k = len(samples)
    return {
        "mean_us": sum(samples) / k,
        "p50_us": samples[k // 2],
        "p95_us": samples[int(k * 0.95)],
        "p99_us": samples[int(k * 0.99)],
        "max_us": samples[-1],
        "n": k,
    }


# ── Registry microbenchmarks ──────────────────────────────────────────────────

def bench_fingerprint(tokens: List[int], n: int = 1000) -> Dict[str, float]:
    """Keyed fingerprint (SHA-256 of model+tokenizer+tokens)."""
    return _bench(
        lambda: PublicRegistry.fingerprint(MODEL_ID, TOKENIZER, tokens),
        n=n,
    )


def bench_issue(registry: PublicRegistry, tokens: List[int], n: int = 1000) -> Dict[str, float]:
    """Issue (sign) a new authorization including HMAC computation."""
    i = [0]

    def _issue():
        i[0] += 1
        registry.issue(
            public_object_id=f"bench-obj-{i[0]}",
            issuer="exp6",
            model_id=MODEL_ID,
            tokenizer_version=TOKENIZER,
            token_ids=tokens,
            expires_at=time.time() + 7200,
        )

    return _bench(_issue, n=n)


def bench_install(registry: PublicRegistry, auth: PublicAuthorization,
                   tokens: List[int], n: int = 1000) -> Dict[str, float]:
    """Install (verify + register) an externally issued authorization."""
    return _bench(
        lambda: registry.install(auth, tokens),
        n=n,
    )


def bench_verify(registry: PublicRegistry, auth: PublicAuthorization,
                  tokens: List[int], n: int = 1000) -> Dict[str, float]:
    """Verify a fully installed authorization (all checks pass)."""
    return _bench(
        lambda: registry.verify(auth, MODEL_ID, TOKENIZER, tokens),
        n=n,
    )


def bench_revoke(n: int = 500) -> Dict[str, float]:
    """Revoke (mark as revoked) an installed object."""
    def _revoke():
        reg = PublicRegistry(OPERATOR_KEY, policy_epoch=POLICY_EPOCH)
        tokens = list(range(32, 96))
        auth = reg.issue("obj-r", "exp6", MODEL_ID, TOKENIZER, tokens,
                         time.time() + 3600)
        reg.revoke(auth.public_object_id)

    return _bench(_revoke, n=n)


def bench_registry_verify_vs_size(
    sizes: List[int], tokens: List[int], n: int = 500
) -> Dict[int, Dict[str, float]]:
    """Verify latency as registry grows from 1 to max_size entries."""
    results = {}
    for size in sizes:
        reg = PublicRegistry(OPERATOR_KEY, policy_epoch=POLICY_EPOCH)
        # Fill registry with 'size' entries.
        auths = []
        for i in range(size):
            t = list(range(i * 2, i * 2 + 32))
            a = reg.issue(f"obj-{i}", "exp6", MODEL_ID, TOKENIZER, t,
                          time.time() + 7200)
            auths.append((a, t))
        # Measure verify for the last entry (installed).
        last_auth, last_tokens = auths[-1]
        stats = _bench(
            lambda: reg.verify(last_auth, MODEL_ID, TOKENIZER, last_tokens),
            n=n,
        )
        results[size] = stats
    return results


# ── Ledger microbenchmarks ────────────────────────────────────────────────────

def bench_ledger_add_hits_memory(n: int = 2000) -> Dict[str, float]:
    """DurableLedger.add_hits with in-memory ledger (no fsync)."""
    ledger = DurableLedger(path=None)
    i = [0]

    def _add():
        i[0] += 1
        ledger.add_hits(f"fp-{i[0] % 100}", 1, persist=False)

    return _bench(_add, n=n)


def bench_ledger_flush(tmp_dir: str, n: int = 100) -> Dict[str, float]:
    """DurableLedger.flush (write + fsync a 10-entry ledger file)."""
    path = os.path.join(tmp_dir, "bench_ledger.json")
    ledger = DurableLedger(path=path)
    for i in range(10):
        ledger.add_hits(f"fp-{i}", i * 3, persist=False)

    return _bench(ledger.flush, n=n)


def bench_ledger_charged_hits(n: int = 2000) -> Dict[str, float]:
    """DurableLedger.charged_hits lookup (in-memory)."""
    ledger = DurableLedger(path=None)
    for i in range(100):
        ledger.add_hits(f"fp-{i}", i + 1, persist=False)
    return _bench(lambda: ledger.charged_hits("fp-50"), n=n)


def bench_ledger_load(tmp_dir: str, n: int = 100) -> Dict[str, float]:
    """DurableLedger cold-load (deserialize + validate) for a 100-entry file."""
    path = os.path.join(tmp_dir, "bench_ledger_load.json")
    # Create a 100-entry ledger file.
    ledger0 = DurableLedger(path=path)
    for i in range(100):
        ledger0.add_hits(f"fp-{i}", i + 1, persist=False)
    ledger0.flush()

    def _load():
        DurableLedger(path=path)  # constructs and loads

    return _bench(_load, n=n)


# ── Metadata size estimates ───────────────────────────────────────────────────

def estimate_metadata_sizes(auth: PublicAuthorization) -> Dict[str, int]:
    """Rough in-memory sizes (bytes) of core SafeKV metadata objects."""
    import json as _json
    import sys as _sys

    def _json_size(obj) -> int:
        """Serialize to JSON and return byte length as a stable size estimate."""
        return len(_json.dumps(obj, default=str).encode("utf-8"))

    def _auth_size(a: PublicAuthorization) -> int:
        return _json_size(a.to_dict())

    def _registry_size(entries: int) -> int:
        reg = PublicRegistry(OPERATOR_KEY, policy_epoch=POLICY_EPOCH)
        objs = []
        for i in range(entries):
            t = list(range(i * 2, i * 2 + 16))
            a = reg.issue(f"obj-sz-{i}", "exp6", MODEL_ID, TOKENIZER, t,
                          time.time() + 7200)
            objs.append(a.to_dict())
        return _json_size({
            "policy_epoch": POLICY_EPOCH,
            "authorizations": {f"obj-sz-{i}": objs[i] for i in range(entries)},
            "revoked_ids": [],
        })

    def _ledger_size(entries: int) -> int:
        store = {f"fingerprint-{i:064x}": i for i in range(entries)}
        return _json_size(store)

    return {
        "PublicAuthorization_json_bytes": _auth_size(auth),
        "PublicRegistry_1_entry_json_bytes": _registry_size(1),
        "PublicRegistry_100_entry_json_bytes": _registry_size(100),
        "DurableLedger_100_entry_json_bytes": _ledger_size(100),
        "DurableLedger_1000_entry_json_bytes": _ledger_size(1000),
    }


def _build_ledger_100() -> DurableLedger:
    ledger = DurableLedger(path=None)
    for i in range(100):
        ledger.add_hits(f"fingerprint-{i:064x}", i, persist=False)
    return ledger


# ── Main ──────────────────────────────────────────────────────────────────────

def fmt(stats: Dict[str, float]) -> str:
    return (
        f"mean={stats['mean_us']:.2f}µs  "
        f"p50={stats['p50_us']:.2f}µs  "
        f"p95={stats['p95_us']:.2f}µs  "
        f"p99={stats['p99_us']:.2f}µs  "
        f"n={stats['n']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "exp6_registry_microbench.json",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument(
        "--registry-sizes",
        nargs="+",
        type=int,
        default=[1, 10, 100, 1000],
        help="Registry sizes for verify-throughput scaling test",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokens = list(range(100, 164))  # 64 token IDs
    registry = PublicRegistry(OPERATOR_KEY, policy_epoch=POLICY_EPOCH)
    ref_auth = registry.issue("ref-obj", "exp6", MODEL_ID, TOKENIZER, tokens,
                               time.time() + 7200)
    registry.install(ref_auth, tokens)

    results: Dict[str, object] = {}

    print("\n=== SafeKV Exp #6 – Registry & Ledger Microbenchmarks ===\n")

    # ── Registry ops ─────────────────────────────────────────────────────────
    n = args.n_samples

    print("1. Fingerprint (SHA-256 of model+tokenizer+tokens)")
    s = bench_fingerprint(tokens, n=n)
    results["fingerprint"] = s
    print(f"   {fmt(s)}")

    print("2. Issue (sign) authorization")
    fresh_reg = PublicRegistry(OPERATOR_KEY, policy_epoch=POLICY_EPOCH)
    s = bench_issue(fresh_reg, tokens, n=min(n, 500))
    results["issue"] = s
    print(f"   {fmt(s)}")

    print("3. Install (verify + register) authorization")
    install_reg = PublicRegistry(OPERATOR_KEY, policy_epoch=POLICY_EPOCH)
    install_auth = install_reg.issue("inst-obj", "exp6", MODEL_ID, TOKENIZER,
                                      tokens, time.time() + 7200)
    s = bench_install(install_reg, install_auth, tokens, n=n)
    results["install"] = s
    print(f"   {fmt(s)}")

    print("4. Verify installed authorization (fast path)")
    verify_reg = PublicRegistry(OPERATOR_KEY, policy_epoch=POLICY_EPOCH)
    verify_auth = verify_reg.issue("ver-obj", "exp6", MODEL_ID, TOKENIZER,
                                    tokens, time.time() + 7200)
    verify_reg.install(verify_auth, tokens)
    s = bench_verify(verify_reg, verify_auth, tokens, n=n)
    results["verify"] = s
    print(f"   {fmt(s)}")

    print("5. Revoke (remove from registry + add to revoked set)")
    s = bench_revoke(n=min(n, 500))
    results["revoke"] = s
    print(f"   {fmt(s)}")

    print("6. Verify throughput vs registry size")
    size_results = bench_registry_verify_vs_size(args.registry_sizes, tokens, n=min(n, 500))
    results["verify_vs_size"] = {str(k): v for k, v in size_results.items()}
    for size, s in size_results.items():
        print(f"   size={size:>4}: {fmt(s)}")

    # ── Ledger ops ────────────────────────────────────────────────────────────
    print("\n7. DurableLedger.add_hits (in-memory, no fsync)")
    s = bench_ledger_add_hits_memory(n=min(n * 2, 2000))
    results["ledger_add_hits_memory"] = s
    print(f"   {fmt(s)}")

    with tempfile.TemporaryDirectory() as tmp:
        print("8. DurableLedger.flush (fsync 10-entry file)")
        s = bench_ledger_flush(tmp, n=min(n // 10, 100))
        results["ledger_flush"] = s
        print(f"   {fmt(s)}")

        print("9. DurableLedger.charged_hits (in-memory lookup)")
        s = bench_ledger_charged_hits(n=min(n * 2, 2000))
        results["ledger_charged_hits"] = s
        print(f"   {fmt(s)}")

        print("10. DurableLedger cold load (100-entry file)")
        s = bench_ledger_load(tmp, n=min(n // 10, 100))
        results["ledger_cold_load"] = s
        print(f"   {fmt(s)}")

    # ── Metadata sizes ────────────────────────────────────────────────────────
    print("\n11. Metadata sizes (pickle estimate)")
    sizes = estimate_metadata_sizes(ref_auth)
    results["metadata_sizes"] = sizes
    for k, v in sizes.items():
        print(f"   {k}: {v} bytes")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {args.output}")

    # Print LaTeX table row.
    print("\n=== Paper table rows (median / P99) ===")
    for label, key in [
        ("Fingerprint",        "fingerprint"),
        ("Issue (sign)",       "issue"),
        ("Install",            "install"),
        ("Verify",             "verify"),
        ("Revoke",             "revoke"),
        ("Ledger add (mem)",   "ledger_add_hits_memory"),
        ("Ledger flush",       "ledger_flush"),
        ("Ledger load",        "ledger_cold_load"),
    ]:
        s = results[key]
        print(f"  {label:<22} & {s['p50_us']:.1f} & {s['p99_us']:.1f} \\\\")


if __name__ == "__main__":
    main()
