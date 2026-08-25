#!/usr/bin/env python3
"""Live two-phase trust-boundary experiment for authenticated principals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import requests


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "submission_gap_experiments"
    / "e4_principal_binding"
)
PROMPT = (
    "SafeKV principal-binding challenge prefix: "
    "the violet lighthouse registry entry belongs to the victim namespace. "
) * 12


def bearer(token: Optional[str]) -> Dict[str, str]:
    return {} if token is None else {"Authorization": f"Bearer {token}"}


def cached_tokens(response: requests.Response) -> Optional[int]:
    body = response.json()
    value = body.get("meta_info", {}).get("cached_tokens")
    return value if isinstance(value, int) else None


def flush(server: str) -> None:
    # Warmup or in-flight requests can make flush return 400 briefly.
    last_error = None
    for _ in range(20):
        response = requests.post(f"{server}/flush_cache", timeout=30)
        if response.status_code == 200:
            time.sleep(0.3)
            return
        last_error = response
        time.sleep(0.5)
    if last_error is not None:
        last_error.raise_for_status()
    raise RuntimeError(f"flush_cache failed on {server}")


def native_generate(
    server: str,
    supplied_user_id: str,
    token: Optional[str],
    *,
    expect_success: bool = True,
) -> requests.Response:
    response = requests.post(
        f"{server}/generate",
        headers=bearer(token),
        json={
            "text": PROMPT,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
                "user_id": supplied_user_id,
            },
        },
        timeout=180,
    )
    if expect_success:
        response.raise_for_status()
    return response


def rejection_matrix(server: str) -> Dict[str, int]:
    native_body = {
        "text": PROMPT,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "user_id": "victim",
        },
    }
    openai_body = {
        "model": "phi4",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 1,
        "user_id": "victim",
    }
    cases = {}
    for api, path, body in (
        ("native", "/generate", native_body),
        ("openai", "/v1/chat/completions", openai_body),
    ):
        for credential, token in (("missing", None), ("unknown", "unknown-token")):
            response = requests.post(
                f"{server}{path}",
                headers=bearer(token),
                json=body,
                timeout=30,
            )
            cases[f"{api}_{credential}"] = response.status_code
    return cases


def load_tokens(path: Path) -> Dict[str, str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    principal_to_token = {principal: token for token, principal in mapping.items()}
    return {
        "victim": principal_to_token["victim"],
        "attacker": principal_to_token["attacker"],
    }


def run_phase(
    server: str, phase: str, credentials: Optional[Path], output_dir: Path
) -> dict:
    tokens = load_tokens(credentials) if credentials else {}
    victim_token = tokens.get("victim")
    attacker_token = tokens.get("attacker")
    victim_supplied_id = "victim" if phase == "disabled" else "attacker"

    flush(server)
    native_generate(server, victim_supplied_id, victim_token)
    spoof = native_generate(server, "victim", attacker_token)

    flush(server)
    native_generate(server, victim_supplied_id, victim_token)
    honest = native_generate(server, "attacker", attacker_token)

    effective_endpoint = requests.get(
        f"{server}/safekv/effective_principal",
        headers=bearer(attacker_token),
        timeout=30,
    )
    result = {
        "phase": phase,
        "binding_enabled": phase == "enabled",
        "attacker_supplied_user_id_for_spoof": "victim",
        "spoof_cached_tokens": cached_tokens(spoof),
        "honest_attacker_cached_tokens": cached_tokens(honest),
        "cache_hit_evidence_available": cached_tokens(spoof) is not None,
        "effective_principal_endpoint_status": effective_endpoint.status_code,
        "effective_principal": (
            effective_endpoint.json().get("effective_principal")
            if effective_endpoint.status_code == 200
            else None
        ),
    }

    if phase == "disabled":
        result["negative_control_pass"] = (
            result["cache_hit_evidence_available"]
            and result["spoof_cached_tokens"] > result["honest_attacker_cached_tokens"]
            and effective_endpoint.status_code == 404
        )
    else:
        result["rejections"] = rejection_matrix(server)
        result["positive_control_pass"] = (
            all(status == 401 for status in result["rejections"].values())
            and result["effective_principal"] == "attacker"
            and (
                not result["cache_hit_evidence_available"]
                or result["spoof_cached_tokens"]
                <= result["honest_attacker_cached_tokens"]
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{phase}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return result


def finalize(output_dir: Path) -> None:
    phases = {}
    missing = []
    for name in ("disabled", "enabled"):
        path = output_dir / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        phases[name] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise FileNotFoundError(
            f"missing phase result(s): {', '.join(missing)} under {output_dir}"
        )
    manifest = {
        "experiment": "e4_principal_binding",
        "backbone": "phi4",
        "safekv_mode": "strict",
        "phases": phases,
        "model_scope": (
            "Principal authentication and namespace selection occur before model "
            "scheduling and cache lookup; this trust-boundary behavior is "
            "model-independent, so one backbone is sufficient."
        ),
        "tokens_recorded": False,
        "negative_control_pass": phases["disabled"].get("negative_control_pass"),
        "positive_control_pass": phases["enabled"].get("positive_control_pass"),
    }
    path = output_dir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8094")
    parser.add_argument("--phase", choices=["disabled", "enabled"])
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.finalize:
        finalize(args.output_dir)
        return
    if args.phase is None:
        parser.error("--phase is required unless --finalize is used")
    if args.phase == "enabled" and args.credentials is None:
        parser.error("--credentials is required for the enabled phase")
    run_phase(args.server.rstrip("/"), args.phase, args.credentials, args.output_dir)


if __name__ == "__main__":
    main()
