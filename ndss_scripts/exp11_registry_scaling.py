#!/usr/bin/env python3
"""SafeKV Exp #11 (appendix) – Public registry and working-set scalability.

Paper: supplementary evaluation §A.4
  "Public working-set scalability"

CPU-only portion: sweeps registry entries (1 to 10000) and measures
  - registry verify latency
  - manifest install time (batch install N entries)
  - ledger memory footprint
  - revocation cost

Live-server portion (requires --server): sweeps pinned Public KV footprint
  sizes and measures TTFT / throughput.

Usage:
  # CPU-only benchmarks:
  python exp11_registry_scaling.py --output results/exp11/scaling.json

  # With live server:
  python exp11_registry_scaling.py --server http://127.0.0.1:8092 \
      --model phi4 --model-path /path/to/phi4 \
      --output results/exp11/scaling_live.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional

# ── Stubs ─────────────────────────────────────────────────────────────────────
_pool_stub = types.ModuleType("sglang.srt.mem_cache.memory_pool")
_pool_stub.ReqToTokenPool = object
_pool_stub.TokenToKVPoolAllocator = object
sys.modules["sglang.srt.mem_cache.memory_pool"] = _pool_stub

from sglang.srt.mem_cache.safekv_policy import DurableLedger, PublicRegistry  # noqa: E402

OPERATOR_KEY = b"safekv-exp11-key"
MODEL_ID = "phi-4-14b"
TOKENIZER = "phi-4-14b-tokenizer"


def _bench(fn, n: int) -> Dict[str, float]:
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
    }


def bench_verify_vs_size(sizes: List[int], n_samples: int = 300) -> Dict:
    results = {}
    for size in sizes:
        reg = PublicRegistry(OPERATOR_KEY, policy_epoch=1)
        auths = []
        for i in range(size):
            t = list(range(i * 2, i * 2 + 32))
            a = reg.issue(f"obj-{i}", "ctrl", MODEL_ID, TOKENIZER, t, time.time() + 7200)
            auths.append((a, t))
        last_auth, last_tokens = auths[-1]
        stats = _bench(
            lambda: reg.verify(last_auth, MODEL_ID, TOKENIZER, last_tokens),
            n=n_samples,
        )
        results[size] = stats
        print(f"  size={size:>5}: p50={stats['p50_us']:.1f}µs  p99={stats['p99_us']:.1f}µs")
    return results


def bench_batch_install(sizes: List[int], n_samples: int = 10) -> Dict:
    """Measure time to batch-install N authorizations into a fresh registry."""
    results = {}
    for size in sizes:
        # Pre-issue authorizations with a separate registry.
        issuer = PublicRegistry(OPERATOR_KEY, policy_epoch=1)
        auths_with_tokens = []
        for i in range(size):
            t = list(range(i * 2, i * 2 + 32))
            a = issuer.issue(f"obj-{i}", "ctrl", MODEL_ID, TOKENIZER, t, time.time() + 7200)
            auths_with_tokens.append((a, t))

        def _batch_install():
            reg = PublicRegistry(OPERATOR_KEY, policy_epoch=1)
            for auth, tokens in auths_with_tokens:
                reg.install(auth, tokens)

        elapsed_ms = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            _batch_install()
            elapsed_ms.append((time.perf_counter() - t0) * 1000)

        elapsed_ms.sort()
        results[size] = {
            "mean_ms": sum(elapsed_ms) / len(elapsed_ms),
            "p50_ms": elapsed_ms[len(elapsed_ms) // 2],
            "per_entry_us": sum(elapsed_ms) / len(elapsed_ms) / size * 1000,
        }
        print(
            f"  size={size:>5}: batch={results[size]['mean_ms']:.1f}ms  "
            f"per_entry={results[size]['per_entry_us']:.1f}µs"
        )
    return results


def bench_revocation_vs_size(sizes: List[int], n_samples: int = 100) -> Dict:
    results = {}
    for size in sizes:
        samples = []
        for sample in range(n_samples):
            reg = PublicRegistry(OPERATOR_KEY, policy_epoch=1)
            for i in range(size):
                t = list(range(i * 2, i * 2 + 32))
                reg.issue(
                    f"obj-{sample}-{i}",
                    "ctrl",
                    MODEL_ID,
                    TOKENIZER,
                    t,
                    time.time() + 7200,
                )
            t0 = time.perf_counter()
            existed = reg.revoke(f"obj-{sample}-0")
            samples.append((time.perf_counter() - t0) * 1e6)
            if not existed:
                raise AssertionError("revocation benchmark revoked no object")
        samples.sort()
        stats = {
            "mean_us": sum(samples) / len(samples),
            "p50_us": samples[len(samples) // 2],
            "p95_us": samples[int(len(samples) * 0.95)],
            "p99_us": samples[int(len(samples) * 0.99)],
        }
        results[size] = stats
        print(f"  size={size:>5}: p50={stats['p50_us']:.1f}µs  p99={stats['p99_us']:.1f}µs")
    return results


def bench_ledger_footprint(entry_counts: List[int]) -> Dict:
    """Memory footprint (bytes) of ledger JSON for N entries."""
    results = {}
    for n in entry_counts:
        ledger = DurableLedger(path=None)
        for i in range(n):
            ledger.add_hits(f"fp-{i:064x}", i, persist=False)
        snap = ledger.snapshot()
        json_bytes = len(json.dumps(dict(snap)).encode("utf-8"))
        results[n] = json_bytes
        print(f"  entries={n:>6}: {json_bytes} bytes ({json_bytes/1024:.1f} KB)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "exp11_registry_scaling.json",
    )
    parser.add_argument(
        "--registry-sizes",
        nargs="+",
        type=int,
        default=[1, 10, 100, 500, 1000, 5000, 10000],
    )
    parser.add_argument(
        "--ledger-sizes",
        nargs="+",
        type=int,
        default=[100, 1000, 10000, 100000],
    )
    parser.add_argument("--n-samples", type=int, default=300)
    # Optional live server measurements.
    parser.add_argument("--server", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--public-token-levels",
        nargs="+",
        type=int,
        default=[0, 512, 2048, 8192],
    )
    parser.add_argument("--live-requests", type=int, default=20)
    args = parser.parse_args()

    results: Dict = {}

    print("\n=== SafeKV Exp #11 – Registry & Working-Set Scalability ===\n")

    print("1. Registry verify latency vs. size")
    results["verify_vs_size"] = bench_verify_vs_size(
        args.registry_sizes, n_samples=min(args.n_samples, 300)
    )

    print("\n2. Batch manifest install time vs. size")
    results["batch_install_vs_size"] = bench_batch_install(
        [s for s in args.registry_sizes if s <= 1000], n_samples=10
    )

    print("\n3. Revocation cost vs. registry size")
    results["revocation_vs_size"] = bench_revocation_vs_size(
        [s for s in args.registry_sizes if s <= 1000], n_samples=min(args.n_samples, 100)
    )

    print("\n4. Ledger memory footprint vs. entry count")
    results["ledger_footprint"] = bench_ledger_footprint(args.ledger_sizes)

    # Optional: live server working-set measurements.
    if args.server and args.model and args.model_path:
        import requests
        from transformers import AutoConfig, AutoTokenizer

        print("\n5. Live server: pinned Public KV working-set sweep")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        layers = int(getattr(config, "num_hidden_layers"))
        attention_heads = int(getattr(config, "num_attention_heads"))
        kv_heads = int(
            getattr(config, "num_key_value_heads", attention_heads)
        )
        head_dim = int(
            getattr(
                config,
                "head_dim",
                int(getattr(config, "hidden_size")) // attention_heads,
            )
        )
        kv_bytes_per_token = 2 * layers * kv_heads * head_dim * 2
        vocab_size = int(getattr(config, "vocab_size"))
        live_results = {}

        def flush_server() -> None:
            response = requests.post(
                f"{args.server.rstrip('/')}/flush_cache", timeout=30
            )
            if response.status_code not in (200, 400):
                response.raise_for_status()
            time.sleep(0.5)
            if snapshot()["variants"]:
                raise RuntimeError("cache did not empty before P11 level")

        def snapshot() -> Dict:
            response = requests.get(
                f"{args.server.rstrip('/')}/get_server_info", timeout=30
            )
            response.raise_for_status()
            states = response.json()["internal_states"]
            if len(states) != 1:
                raise RuntimeError("P11 live sweep requires dp_size=1")
            return states[0]["safekv"]

        def generate(
            token_ids: List[int],
            user_id: str,
            max_new_tokens: int,
            authorization=None,
        ) -> tuple[float, int]:
            params = {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "user_id": user_id,
            }
            if authorization is not None:
                params["safekv_public_authorization"] = (
                    authorization.to_dict()
                )
            t0 = time.perf_counter()
            response = requests.post(
                f"{args.server.rstrip('/')}/generate",
                json={
                    "input_ids": token_ids,
                    "sampling_params": params,
                    "stream": False,
                },
                timeout=300,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            response.raise_for_status()
            body = response.json()
            output_tokens = len(
                body.get("token_ids") or body.get("output_ids") or []
            )
            return elapsed_ms, output_tokens or max_new_tokens

        object_tokens = 128
        safe_vocab = max(1024, vocab_size - 1024)
        for target_public_tokens in args.public_token_levels:
            from sglang.srt.mem_cache.safekv_policy import PublicRegistry as _Reg
            import os as _os

            key = _os.environ.get("SAFEKV_OPERATOR_KEY", "safekv-exp11-key").encode()
            reg = _Reg(key, policy_epoch=1)
            tokenizer_version = requests.get(
                f"{args.server.rstrip('/')}/get_model_info", timeout=30
            ).json()["tokenizer_path"]

            flush_server()
            n_public_objects = (
                target_public_tokens + object_tokens - 1
            ) // object_tokens
            public_prefixes = []
            prewarm_ms = []
            first_auth = None
            for i in range(n_public_objects):
                remaining = target_public_tokens - i * object_tokens
                length = min(object_tokens, remaining)
                text_ids = [
                    100 + ((i * object_tokens + j) % safe_vocab)
                    for j in range(length)
                ]
                auth = reg.issue(
                    f"exp11-pub-{i}", "ctrl", args.model, tokenizer_version, text_ids,
                    time.time() + 86400,
                )
                elapsed, _ = generate(
                    text_ids,
                    "operator-prewarm",
                    1,
                    authorization=auth,
                )
                prewarm_ms.append(elapsed)
                public_prefixes.append(text_ids)
                if first_auth is None:
                    first_auth = auth

            after_prewarm = snapshot()
            resident_public_tokens = sum(
                int(variant["token_count"])
                for variant in after_prewarm["variants"]
                if variant["namespace_visibility"] == "verified_public"
            )

            no_victim_ttft = None
            after_victim_ttft = None
            refresh_ms = None
            if public_prefixes:
                no_victim_ttft, _ = generate(
                    public_prefixes[0], "public-probe-before-victim", 1
                )
                generate(public_prefixes[0], "victim-use", 1)
                after_victim_ttft, _ = generate(
                    public_prefixes[0], "public-probe-after-victim", 1
                )
                refresh_ms, _ = generate(
                    public_prefixes[0],
                    "operator-refresh",
                    1,
                    authorization=first_auth,
                )

            private_latencies = []
            private_output_tokens = 0
            private_prefixes = []
            workload_start = time.perf_counter()
            private_inserted_tokens = 0
            for i in range(args.live_requests):
                private_ids = [
                    100
                    + (
                        (
                            20000
                            + target_public_tokens
                            + i * 96
                            + j
                        )
                        % safe_vocab
                    )
                    for j in range(96)
                ]
                elapsed, output_count = generate(
                    private_ids, f"private-{i}", 64
                )
                private_latencies.append(elapsed)
                private_output_tokens += output_count
                private_inserted_tokens += len(private_ids)
                private_prefixes.append(private_ids)
            workload_elapsed = time.perf_counter() - workload_start

            final_snapshot = snapshot()
            private_resident_tokens = sum(
                int(variant["token_count"])
                for variant in final_snapshot["variants"]
                if variant["namespace_visibility"] == "private"
            )
            public_objects_resident = len(
                final_snapshot["public_object_ids"]
            )
            private_reuse_hits = 0
            for i, private_ids in enumerate(private_prefixes):
                generate(private_ids, f"private-{i}", 1)
                probe_snapshot = snapshot()
                lookup_events = [
                    event
                    for event in probe_snapshot["events"]
                    if event["name"] == "lookup"
                    and event["attributes"].get("requester")
                    == f"private-{i}"
                ]
                if (
                    lookup_events
                    and lookup_events[-1]["attributes"].get("hit")
                ):
                    private_reuse_hits += 1
            live_results[target_public_tokens] = {
                "target_public_tokens": target_public_tokens,
                "public_objects": n_public_objects,
                "resident_public_tokens": resident_public_tokens,
                "estimated_public_kv_mb": (
                    resident_public_tokens * kv_bytes_per_token / 1e6
                ),
                "prewarm_total_ms": sum(prewarm_ms),
                "prewarm_per_object_p50_ms": (
                    statistics.median(prewarm_ms) if prewarm_ms else 0
                ),
                "no_victim_use_probe_ms": no_victim_ttft,
                "after_victim_use_probe_ms": after_victim_ttft,
                "refresh_ms": refresh_ms,
                "private_latency_p50_ms": statistics.median(
                    private_latencies
                ),
                "private_latency_p95_ms": sorted(private_latencies)[
                    min(
                        int(len(private_latencies) * 0.95),
                        len(private_latencies) - 1,
                    )
                ],
                "throughput_tok_s": (
                    private_output_tokens / workload_elapsed
                ),
                "private_inserted_tokens": private_inserted_tokens,
                "private_resident_tokens": private_resident_tokens,
                "private_retention_ratio": (
                    private_resident_tokens / private_inserted_tokens
                ),
                "private_reuse_hit_rate": (
                    private_reuse_hits / len(private_prefixes)
                ),
                "public_objects_resident_after_private_load": (
                    public_objects_resident
                ),
            }
            print(
                f"  pub_tokens={target_public_tokens:>5}: "
                f"KV={live_results[target_public_tokens]['estimated_public_kv_mb']:.1f}MB "
                f"private_p50={live_results[target_public_tokens]['private_latency_p50_ms']:.1f}ms "
                f"tput={live_results[target_public_tokens]['throughput_tok_s']:.1f}tok/s "
                f"private_hit={live_results[target_public_tokens]['private_reuse_hit_rate']:.2f}",
                flush=True,
            )

        results["live_ttft_vs_public_objs"] = live_results
        results["kv_bytes_per_token"] = kv_bytes_per_token

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved → {args.output}")

    # Print LaTeX summary rows.
    print("\n=== Verify latency summary (p50 / p99 µs) ===")
    for size, s in results["verify_vs_size"].items():
        print(f"  {size:>5} entries & {s['p50_us']:.1f} & {s['p99_us']:.1f} \\\\")


if __name__ == "__main__":
    main()
