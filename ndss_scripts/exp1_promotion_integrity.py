#!/usr/bin/env python3
"""Run SafeKV Exp #1 promotion-integrity and namespace invariants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from transformers import AutoTokenizer

from sglang.srt.mem_cache.safekv_policy import PublicAuthorization, PublicRegistry


ATTACKER_COUNTS = (1, 2, 4, 8)
AUTH_CONDITIONS = ("none", "forged", "stale", "revoked", "valid")
SCHEDULES = ("sequential", "concurrent")
CSV_FIELDS = (
    "model",
    "trial_id",
    "seed",
    "A",
    "auth",
    "schedule",
    "prefix_id",
    "prefix_token_length",
    "unauth_public_promotions",
    "victim_node_relabels",
    "private_address_aliases",
    "cross_tenant_private_hits",
    "public_object_created",
    "public_reuses_victim_kv",
    "victim_private_still_owned",
    "attacker_hit_via_public",
    "attacker_hit_via_private",
    "auth_verify_result",
    "pass",
    "error",
)


def load_prefixes(dataset: Path, tokenizer, count: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    texts = []
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = item.get("source_text") or item.get("text")
            if text and len(text.strip()) >= 40:
                texts.append(text.strip())
    if not texts:
        raise RuntimeError("no usable sensitive prefixes in dataset")
    rng.shuffle(texts)
    prefixes = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        token_ids = token_ids[:128]
        if len(token_ids) >= 16:
            prefixes.append(token_ids)
        if len(prefixes) >= count:
            break
    if len(prefixes) < count:
        raise RuntimeError(f"need {count} prefixes, found {len(prefixes)}")
    return prefixes


class ExperimentClient:
    def __init__(self, server: str, timeout: float = 120.0):
        self.server = server.rstrip("/")
        self.timeout = timeout

    def flush(self) -> None:
        response = requests.post(
            f"{self.server}/flush_cache", timeout=self.timeout
        )
        response.raise_for_status()
        for _ in range(100):
            snapshot = self.snapshot()
            if not snapshot["variants"]:
                return
            time.sleep(0.05)
        raise RuntimeError("cache did not become empty after flush")

    def snapshot(self) -> Dict[str, object]:
        response = requests.get(
            f"{self.server}/get_server_info", timeout=self.timeout
        )
        response.raise_for_status()
        states = response.json()["internal_states"]
        if len(states) != 1:
            raise RuntimeError("Exp #1 requires dp_size=1")
        return states[0]["safekv"]

    def generate(
        self,
        token_ids: List[int],
        principal: str,
        authorization: Optional[PublicAuthorization] = None,
    ) -> Dict[str, object]:
        sampling_params: Dict[str, object] = {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "user_id": principal,
        }
        if authorization is not None:
            sampling_params["safekv_public_authorization"] = (
                authorization.to_dict()
            )
        response = requests.post(
            f"{self.server}/generate",
            json={
                "input_ids": token_ids,
                "sampling_params": sampling_params,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def make_authorization(
    condition: str,
    tokens: Iterable[int],
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    object_id: str,
) -> Optional[PublicAuthorization]:
    if condition == "none":
        return None
    key = operator_key.encode("utf-8")
    epoch = 1
    revoked = False
    if condition == "forged":
        key = b"attacker-controlled-key"
    elif condition == "stale":
        epoch = 0
    elif condition == "revoked":
        revoked = True
    signer = PublicRegistry(key, policy_epoch=epoch)
    return signer.issue(
        public_object_id=object_id,
        issuer="safekv-exp1-control-plane",
        model_id=model_id,
        tokenizer_version=tokenizer_version,
        token_ids=tokens,
        expires_at=time.time() + 3600,
        revoked=revoked,
    )


def release_concurrently(tasks: List[Tuple], client: ExperimentClient) -> None:
    barrier = threading.Barrier(len(tasks))

    def invoke(task):
        token_ids, principal, authorization = task
        barrier.wait(timeout=30)
        return client.generate(token_ids, principal, authorization)

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [pool.submit(invoke, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


def identities(snapshot, visibility: str, identity: Optional[str] = None):
    variants = [
        variant
        for variant in snapshot["variants"]
        if variant["namespace_visibility"] == visibility
    ]
    if identity is not None:
        variants = [
            variant
            for variant in variants
            if variant["namespace_identity"] == identity
        ]
    return variants


def address_set(variants) -> set:
    return {
        variant["kv_address_id"]
        for variant in variants
        if variant["kv_address_id"] is not None
    }


def evaluate_trial(
    client: ExperimentClient,
    model: str,
    token_ids: List[int],
    trial_id: int,
    attackers: int,
    auth_condition: str,
    schedule: str,
    model_id: str,
    tokenizer_version: str,
    operator_key: str,
    seed: int,
) -> Dict[str, object]:
    victim = f"victim-{trial_id}"
    attacker_ids = [f"attacker-{trial_id}-{i}" for i in range(attackers)]
    object_id = f"public-{trial_id}-{attackers}-{schedule}"
    # SGLang can reuse at most input_ids[:-1] during the first prefill.
    authorized_tokens = token_ids[:-1]
    authorization = make_authorization(
        auth_condition,
        authorized_tokens,
        model_id,
        tokenizer_version,
        operator_key,
        object_id,
    )
    client.flush()

    if schedule == "sequential":
        client.generate(token_ids, victim)
        if auth_condition == "valid":
            client.generate(token_ids, "operator-prewarm", authorization)
            attacker_auth = None
        else:
            attacker_auth = authorization
        for attacker in attacker_ids:
            client.generate(token_ids, attacker, attacker_auth)
    else:
        task_auth = None if auth_condition == "valid" else authorization
        tasks = [(token_ids, victim, None)] + [
            (token_ids, attacker, task_auth) for attacker in attacker_ids
        ]
        release_concurrently(tasks, client)
        if auth_condition == "valid":
            client.generate(token_ids, "operator-prewarm", authorization)

    if auth_condition == "valid":
        # Positive control: a normal attacker can now use the independently
        # materialized Public object without touching victim-private state.
        client.generate(token_ids, attacker_ids[0])

    snapshot = client.snapshot()
    counters = snapshot["counters"]
    victim_variants = identities(snapshot, "private", victim)
    attacker_variants = []
    for attacker in attacker_ids:
        attacker_variants.extend(identities(snapshot, "private", attacker))
    public_variants = identities(snapshot, "verified_public")
    victim_addresses = address_set(victim_variants)
    attacker_addresses = address_set(attacker_variants)
    public_addresses = address_set(public_variants)
    private_aliases = len(victim_addresses & attacker_addresses)
    public_aliases = int(bool(victim_addresses & public_addresses))
    victim_owned = int(bool(victim_variants))
    victim_relabels = sum(
        1
        for variant in snapshot["variants"]
        if variant.get("creator_id") == victim
        and variant["namespace_identity"] != victim
    )
    attacker_public_hit = int(
        any(
            event["name"] == "lookup"
            and event["attributes"].get("requester") in attacker_ids
            and event["attributes"].get("served_namespace")
            == "verified_public"
            and event["attributes"].get("hit")
            for event in snapshot["events"]
        )
    )
    attacker_private_hit = int(
        any(
            event["name"] == "lookup"
            and event["attributes"].get("requester") in attacker_ids
            and event["attributes"].get("served_namespace") == "private"
            and event["attributes"].get("owner")
            != event["attributes"].get("requester")
            and event["attributes"].get("hit")
            for event in snapshot["events"]
        )
    )
    auth_reasons = [
        event["attributes"].get("reason")
        for event in snapshot["events"]
        if event["name"] == "authorization_verified"
    ]
    public_created = int(counters["public_object_created"])
    row = {
        "model": model,
        "trial_id": trial_id,
        "seed": seed,
        "A": attackers,
        "auth": auth_condition,
        "schedule": schedule,
        "prefix_id": f"prefix-{trial_id}",
        "prefix_token_length": len(authorized_tokens),
        "unauth_public_promotions": int(
            counters["unauth_public_promotions"]
        ),
        "victim_node_relabels": victim_relabels,
        "private_address_aliases": private_aliases,
        "cross_tenant_private_hits": int(
            counters["cross_tenant_private_hits"]
        ),
        "public_object_created": public_created,
        "public_reuses_victim_kv": public_aliases,
        "victim_private_still_owned": victim_owned,
        "attacker_hit_via_public": attacker_public_hit,
        "attacker_hit_via_private": attacker_private_hit,
        "auth_verify_result": "|".join(str(reason) for reason in auth_reasons),
        "error": "",
    }
    zero_fields = (
        "unauth_public_promotions",
        "victim_node_relabels",
        "private_address_aliases",
        "cross_tenant_private_hits",
        "attacker_hit_via_private",
    )
    passed = all(row[field] == 0 for field in zero_fields)
    if auth_condition == "valid":
        passed = (
            passed
            and public_created == 1
            and public_aliases == 0
            and victim_owned == 1
            and attacker_public_hit == 1
        )
    else:
        passed = passed and public_created == 0
    row["pass"] = int(passed)
    return row


def load_completed(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (
                int(row["A"]),
                row["auth"],
                row["schedule"],
                int(row["trial_id"]),
            )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parents[1]
        / "datasets"
        / "english_pii_43k.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--operator-key",
        default=os.environ.get(
            "SAFEKV_OPERATOR_KEY", "safekv-exp1-operator-key"
        ),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    client = ExperimentClient(args.server)
    server_info = requests.get(
        f"{args.server.rstrip('/')}/get_model_info", timeout=30
    ).json()
    tokenizer_version = server_info["tokenizer_path"]
    prefixes = load_prefixes(
        args.dataset, tokenizer, args.trials, args.seed
    )
    completed = load_completed(args.output)
    total = len(ATTACKER_COUNTS) * len(AUTH_CONDITIONS) * len(SCHEDULES) * args.trials
    done = len(completed)

    for attackers in ATTACKER_COUNTS:
        for auth_condition in AUTH_CONDITIONS:
            for schedule in SCHEDULES:
                for trial_id in range(args.trials):
                    key = (attackers, auth_condition, schedule, trial_id)
                    if key in completed:
                        continue
                    try:
                        row = evaluate_trial(
                            client,
                            args.model,
                            prefixes[trial_id],
                            trial_id,
                            attackers,
                            auth_condition,
                            schedule,
                            args.model,
                            tokenizer_version,
                            args.operator_key,
                            args.seed,
                        )
                    except Exception as exc:
                        row = {
                            field: ""
                            for field in CSV_FIELDS
                        }
                        row.update(
                            {
                                "model": args.model,
                                "trial_id": trial_id,
                                "seed": args.seed,
                                "A": attackers,
                                "auth": auth_condition,
                                "schedule": schedule,
                                "prefix_id": f"prefix-{trial_id}",
                                "pass": 0,
                                "error": repr(exc),
                            }
                        )
                    append_row(args.output, row)
                    done += 1
                    print(
                        f"[{done}/{total}] A={attackers} auth={auth_condition} "
                        f"schedule={schedule} trial={trial_id} pass={row['pass']}",
                        flush=True,
                    )
                    if row["error"]:
                        raise RuntimeError(row["error"])


if __name__ == "__main__":
    main()
