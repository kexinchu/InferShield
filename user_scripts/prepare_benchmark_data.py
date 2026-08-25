#!/usr/bin/env python3
"""
Prepare benchmark dataset by combining ShareGPT multi-turn conversations
with PII data from english_pii_43k.jsonl.

Strategy:
  - Sample multi-turn conversations from ShareGPT (>= 4 turns)
  - Sample PII texts from english_pii_43k.jsonl
  - For each conversation, randomly insert a PII text into one of the
    user turns (replacing or appending), creating a realistic scenario
    where PII appears naturally in a multi-turn dialogue.

Output: JSONL file where each line is a conversation with messages array
  {"messages": [...], "pii_turn_idx": N, "pii_type": [...], "has_pii": true}

Usage:
  python3 prepare_benchmark_data.py [--num-samples 200] [--output benchmark_data.jsonl]
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "datasets")

SHAREGPT_PATH = os.path.join(DATASET_DIR, "ShareGPT_V3_unfiltered_cleaned_split.json")
PII_PATH = os.path.join(DATASET_DIR, "english_pii_43k.jsonl")


def load_sharegpt(path: str, min_turns: int = 4, max_turns: int = 20) -> List[List[Dict]]:
    """Load ShareGPT conversations with at least min_turns turns."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    for entry in data:
        conv = entry.get("conversations", [])
        # Filter: must have at least min_turns, alternating human/gpt
        if len(conv) < min_turns:
            continue
        if len(conv) > max_turns:
            conv = conv[:max_turns]
        # Ensure proper alternation and non-empty content
        valid = True
        for i, turn in enumerate(conv):
            if not turn.get("value", "").strip():
                valid = False
                break
        if not valid:
            continue
        conversations.append(conv)

    return conversations


def load_pii_texts(path: str) -> List[Dict]:
    """Load PII texts with their labels."""
    pii_items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            bio_labels = item.get("mbert_bio_labels", [])
            if isinstance(bio_labels, str):
                bio_labels = eval(bio_labels)
            # Extract PII types present
            pii_types = set()
            for label in bio_labels:
                if label != "O":
                    pii_types.add(label.replace("B-", "").replace("I-", ""))
            if pii_types:
                pii_items.append({
                    "text": item["source_text"],
                    "pii_types": list(pii_types),
                })
    return pii_items


def build_messages(conv: List[Dict]) -> List[Dict]:
    """Convert ShareGPT format to OpenAI messages format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in conv:
        role = role_map.get(turn["from"], "user")
        messages.append({"role": role, "content": turn["value"]})
    return messages


def inject_pii_into_conversation(
    conv: List[Dict],
    pii_text: str,
    pii_types: List[str],
    inject_mode: str = "random",
) -> Tuple[List[Dict], int]:
    """
    Inject a PII text into a random user turn in the conversation.

    inject_mode:
      - "replace": replace the user message content entirely
      - "append": append PII text to the user message
      - "random": randomly choose replace or append
    """
    messages = build_messages(conv)

    # Find all user turn indices (excluding the last turn if it's user,
    # since we want a response after the PII turn)
    user_indices = [
        i for i, m in enumerate(messages)
        if m["role"] == "user" and i < len(messages) - 1
    ]

    if not user_indices:
        # Fallback: use any user turn
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]

    if not user_indices:
        return messages, -1

    # Pick a random user turn to inject PII
    inject_idx = random.choice(user_indices)

    if inject_mode == "random":
        inject_mode = random.choice(["replace", "append"])

    if inject_mode == "replace":
        messages[inject_idx]["content"] = pii_text
    else:  # append
        original = messages[inject_idx]["content"]
        # Insert PII naturally
        connectors = [
            f"\n\nBy the way, here's some context: {pii_text}",
            f"\n\nAdditional information: {pii_text}",
            f"\n\n{pii_text}",
            f" Also, {pii_text}",
        ]
        messages[inject_idx]["content"] = original + random.choice(connectors)

    return messages, inject_idx


def create_benchmark_dataset(
    num_samples: int = 200,
    seed: int = 42,
    inject_ratio: float = 0.5,
) -> List[Dict]:
    """
    Create benchmark dataset:
      - inject_ratio fraction of conversations get PII injected
      - remaining are clean ShareGPT conversations (negative samples)
    """
    random.seed(seed)

    print(f"[INFO] Loading ShareGPT from {SHAREGPT_PATH} ...")
    conversations = load_sharegpt(SHAREGPT_PATH, min_turns=4)
    print(f"[INFO] Loaded {len(conversations)} multi-turn conversations (>=4 turns)")

    print(f"[INFO] Loading PII texts from {PII_PATH} ...")
    pii_items = load_pii_texts(PII_PATH)
    print(f"[INFO] Loaded {len(pii_items)} PII texts")

    # Sample conversations
    random.shuffle(conversations)
    sampled_convs = conversations[:num_samples]

    # Sample PII texts
    random.shuffle(pii_items)
    num_pii = int(num_samples * inject_ratio)

    dataset = []

    # PII-injected conversations
    for i in range(num_pii):
        conv = sampled_convs[i]
        pii = pii_items[i % len(pii_items)]

        messages, pii_turn_idx = inject_pii_into_conversation(
            conv, pii["text"], pii["pii_types"], inject_mode="random"
        )

        dataset.append({
            "messages": messages,
            "has_pii": True,
            "pii_turn_idx": pii_turn_idx,
            "pii_types": pii["pii_types"],
            "num_turns": len(messages),
        })

    # Clean conversations (no PII)
    for i in range(num_pii, len(sampled_convs)):
        conv = sampled_convs[i]
        messages = build_messages(conv)

        dataset.append({
            "messages": messages,
            "has_pii": False,
            "pii_turn_idx": -1,
            "pii_types": [],
            "num_turns": len(messages),
        })

    # Shuffle to mix PII and non-PII
    random.shuffle(dataset)

    # Print stats
    pii_count = sum(1 for d in dataset if d["has_pii"])
    turn_counts = [d["num_turns"] for d in dataset]
    print(f"\n[INFO] Dataset stats:")
    print(f"  Total samples:    {len(dataset)}")
    print(f"  With PII:         {pii_count}")
    print(f"  Without PII:      {len(dataset) - pii_count}")
    print(f"  Avg turns:        {sum(turn_counts)/len(turn_counts):.1f}")
    print(f"  Min/Max turns:    {min(turn_counts)}/{max(turn_counts)}")

    if pii_count > 0:
        all_types = defaultdict(int)
        for d in dataset:
            for t in d["pii_types"]:
                all_types[t] += 1
        print(f"  PII types:        {dict(sorted(all_types.items(), key=lambda x: -x[1])[:10])}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark dataset")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Total number of conversation samples")
    parser.add_argument("--inject-ratio", type=float, default=0.5,
                        help="Fraction of conversations with PII injected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: results/benchmark_data.jsonl)")

    args = parser.parse_args()

    if args.output is None:
        out_dir = os.path.join(SCRIPT_DIR, "results")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, "benchmark_data.jsonl")

    dataset = create_benchmark_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        inject_ratio=args.inject_ratio,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[INFO] Saved to {args.output}")
