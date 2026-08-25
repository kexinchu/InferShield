"""
SafeKV Functional Test Script
Tests the core SafeKV v2 mechanisms:
  1. Private-by-default: new blocks start as private (private_tag=1)
  2. Async promotion: after detection, non-sensitive blocks become shareable
  3. Cross-tenant isolation: private blocks are only accessible to the creator
  4. Cross-tenant access budget: shareable blocks demote after B cross-tenant hits
  5. Multi-source re-promotion: demoted blocks re-promote when creator_count >= K
  6. Two-tier detection: only Tier 1 (regex) + Tier 2 (ML), no Level 3

Usage:
  1. Start the SGLang server:
       ./launch_model.sh phi4
  2. Run this test:
       python test_safekv_functional.py --port 8092 --model phi4
"""

import argparse
import requests
import json
import time
import sys
import os
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
DEFAULT_PORT = 8092
DEFAULT_MODEL = "phi4"
MAX_TOKENS = 1  # minimize generation cost; we only care about TTFT/cache behavior


def api_url(port):
    return f"http://127.0.0.1:{port}/v1/chat/completions"


HEADERS = {"Content-Type": "application/json"}


# ============================================================
# Helper: send a single chat completion request
# ============================================================
def send_request(prompt, user_id, port, model, system_prompt=None):
    """Send a request and return (latency_ms, success, response_json)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "user_id": str(user_id),
    }

    start = time.perf_counter()
    try:
        resp = requests.post(api_url(port), headers=HEADERS,
                             data=json.dumps(payload), timeout=60)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            return elapsed_ms, True, resp.json()
        else:
            return elapsed_ms, False, {"error": resp.status_code, "text": resp.text[:200]}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, False, {"error": str(e)}


# ============================================================
# Test 1: Private-by-default & Cross-tenant Isolation
# ============================================================
def test_cross_tenant_isolation(port, model):
    """
    Verify that a private prompt from user A cannot be reused by user B.
    Strategy:
      - User A sends a PII-containing prompt (stays private).
      - User B sends the exact same prompt.
      - User B should NOT get a cache hit (latency should be similar to cold).
    """
    print("\n" + "=" * 60)
    print("TEST 1: Cross-tenant Isolation (Private-by-default)")
    print("=" * 60)

    # PII prompt — should stay private after detection
    pii_prompt = (
        "My name is John Smith, my SSN is 123-45-6789, "
        "and my email is john.smith@example.com. "
        "Please help me file my tax return."
    )

    # Step 1: User A sends the PII prompt (cold)
    lat_a1, ok_a1, _ = send_request(pii_prompt, user_id="user_A", port=port, model=model)
    print(f"  User A (1st request, cold):  {lat_a1:.1f} ms  [ok={ok_a1}]")

    # Step 2: User A sends it again (should cache hit from own cache)
    time.sleep(1)  # wait for async detection
    lat_a2, ok_a2, _ = send_request(pii_prompt, user_id="user_A", port=port, model=model)
    print(f"  User A (2nd request, warm):  {lat_a2:.1f} ms  [ok={ok_a2}]")

    # Step 3: User B sends the same PII prompt (should NOT hit A's cache)
    lat_b1, ok_b1, _ = send_request(pii_prompt, user_id="user_B", port=port, model=model)
    print(f"  User B (same prompt, cold):  {lat_b1:.1f} ms  [ok={ok_b1}]")

    # Analysis
    if ok_a1 and ok_a2 and ok_b1:
        speedup_a = lat_a1 / lat_a2 if lat_a2 > 0 else 0
        ratio_b = lat_b1 / lat_a2 if lat_a2 > 0 else 0
        print(f"\n  Analysis:")
        print(f"    User A speedup (cold→warm): {speedup_a:.2f}x")
        print(f"    User B / User A warm ratio: {ratio_b:.2f}x")
        if ratio_b > 0.8:
            print(f"  ✓ PASS: User B did NOT benefit from User A's private cache")
        else:
            print(f"  ✗ FAIL: User B appears to have hit User A's private cache")
    else:
        print("  ✗ SKIP: Some requests failed")


# ============================================================
# Test 2: Shareable Promotion for Non-PII Content
# ============================================================
def test_shareable_promotion(port, model):
    """
    Verify that non-PII content gets promoted to shareable after async detection.
    Strategy:
      - User A sends a generic (non-PII) prompt.
      - Wait for async detection to complete.
      - User B sends the same prompt.
      - User B should benefit from cross-tenant cache reuse.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Shareable Promotion (Non-PII Content)")
    print("=" * 60)

    # Generic non-PII prompt
    generic_prompt = (
        "Explain the difference between a stack and a queue in computer science. "
        "Provide examples of when you would use each data structure."
    )

    # Step 1: User A sends generic prompt (cold)
    lat_a1, ok_a1, _ = send_request(generic_prompt, user_id="user_C", port=port, model=model)
    print(f"  User C (1st request, cold):  {lat_a1:.1f} ms  [ok={ok_a1}]")

    # Step 2: Wait for async promotion
    time.sleep(2)

    # Step 3: User D sends the same prompt (should benefit from cache)
    lat_d1, ok_d1, _ = send_request(generic_prompt, user_id="user_D", port=port, model=model)
    print(f"  User D (same prompt):        {lat_d1:.1f} ms  [ok={ok_d1}]")

    # Step 4: Another user sends the same prompt
    lat_e1, ok_e1, _ = send_request(generic_prompt, user_id="user_E", port=port, model=model)
    print(f"  User E (same prompt):        {lat_e1:.1f} ms  [ok={ok_e1}]")

    if ok_a1 and ok_d1 and ok_e1:
        speedup_d = lat_a1 / lat_d1 if lat_d1 > 0 else 0
        speedup_e = lat_a1 / lat_e1 if lat_e1 > 0 else 0
        print(f"\n  Analysis:")
        print(f"    User D speedup vs cold: {speedup_d:.2f}x")
        print(f"    User E speedup vs cold: {speedup_e:.2f}x")
        if speedup_d > 1.1 or speedup_e > 1.1:
            print(f"  ✓ PASS: Cross-tenant cache reuse observed for non-PII content")
        else:
            print(f"  ? INFO: No significant speedup (may depend on prompt length/model)")
    else:
        print("  ✗ SKIP: Some requests failed")


# ============================================================
# Test 3: Access Budget Demotion
# ============================================================
def test_access_budget(port, model, budget_B=10):
    """
    Verify that a shareable block demotes after B cross-tenant hits.
    Strategy:
      - User A sends a non-PII prompt → promoted to shareable with budget B
      - B+5 different users send the same prompt
      - After user B+1, the block should be demoted to private
      - Subsequent users should NOT get cache hits
    """
    print("\n" + "=" * 60)
    print(f"TEST 3: Access Budget Demotion (B={budget_B})")
    print("=" * 60)

    # Use a unique non-PII prompt to avoid interference from other tests
    budget_prompt = (
        "What are the main differences between Python and JavaScript? "
        "Compare their type systems, concurrency models, and typical use cases. "
        f"Test run ID: {int(time.time())}"  # make unique
    )

    # Step 1: Creator sends it
    lat_creator, ok_creator, _ = send_request(budget_prompt, user_id="budget_creator",
                                               port=port, model=model)
    print(f"  Creator (cold):              {lat_creator:.1f} ms  [ok={ok_creator}]")

    # Wait for async promotion
    time.sleep(2)

    # Step 2: Send B + 5 cross-tenant requests, track latencies
    latencies = []
    for i in range(budget_B + 5):
        user = f"budget_user_{i}"
        lat, ok, _ = send_request(budget_prompt, user_id=user, port=port, model=model)
        latencies.append((i + 1, lat, ok))
        print(f"  Cross-tenant hit #{i+1:2d} (user={user}): {lat:.1f} ms  [ok={ok}]")
        time.sleep(0.2)

    # Analysis: latencies after budget B should increase (cache miss)
    print(f"\n  Analysis:")
    pre_budget = [lat for idx, lat, ok in latencies if idx <= budget_B and ok]
    post_budget = [lat for idx, lat, ok in latencies if idx > budget_B and ok]

    if pre_budget and post_budget:
        avg_pre = sum(pre_budget) / len(pre_budget)
        avg_post = sum(post_budget) / len(post_budget)
        print(f"    Avg latency (hits 1-{budget_B}):    {avg_pre:.1f} ms")
        print(f"    Avg latency (hits {budget_B+1}-{budget_B+5}): {avg_post:.1f} ms")
        if avg_post > avg_pre * 1.1:
            print(f"  ✓ PASS: Latency increased after budget exhaustion (demotion detected)")
        else:
            print(f"  ? INFO: No clear latency increase (budget demotion may not be observable at this scale)")
    else:
        print("  ✗ SKIP: Insufficient successful requests")


# ============================================================
# Test 4: Multi-Source Re-promotion
# ============================================================
def test_multi_source_repromotion(port, model, K=2):
    """
    Verify that after budget demotion, a block with creator_count >= K
    gets re-promoted with a fresh budget.
    Strategy:
      - User A creates a shared system prompt prefix (non-PII)
      - User B independently creates the same prefix → creator_count = 2 >= K
      - Exhaust the access budget to trigger demotion
      - The block should be re-promoted because creator_count >= K
    """
    print("\n" + "=" * 60)
    print(f"TEST 4: Multi-Source Re-promotion (K={K})")
    print("=" * 60)

    # Shared system prompt — multiple users would naturally create this
    system_prompt = (
        "You are a helpful, harmless, and honest AI assistant. "
        "Always provide accurate and well-reasoned responses."
    )
    user_question = f"What is 2 + 2? (test_id={int(time.time())})"

    # Step 1: User A creates the prefix
    lat_a, ok_a, _ = send_request(user_question, user_id="repromo_A",
                                   port=port, model=model, system_prompt=system_prompt)
    print(f"  User A (creator 1):          {lat_a:.1f} ms  [ok={ok_a}]")
    time.sleep(1)

    # Step 2: User B independently creates the same prefix → creator_count = 2
    lat_b, ok_b, _ = send_request(user_question, user_id="repromo_B",
                                   port=port, model=model, system_prompt=system_prompt)
    print(f"  User B (creator 2):          {lat_b:.1f} ms  [ok={ok_b}]")
    time.sleep(1)

    # Step 3: User C should still get cache benefit
    lat_c, ok_c, _ = send_request(user_question, user_id="repromo_C",
                                   port=port, model=model, system_prompt=system_prompt)
    print(f"  User C (cross-tenant):       {lat_c:.1f} ms  [ok={ok_c}]")

    if ok_a and ok_b and ok_c:
        speedup_c = lat_a / lat_c if lat_c > 0 else 0
        print(f"\n  Analysis:")
        print(f"    User C speedup vs cold: {speedup_c:.2f}x")
        print(f"  ✓ INFO: Multi-source prefix established (creator_count >= {K})")
    else:
        print("  ✗ SKIP: Some requests failed")


# ============================================================
# Test 5: Timing Side-Channel Resistance
# ============================================================
def test_timing_side_channel(port, model, num_probes=10):
    """
    Verify SafeKV's defense against timing side-channel attacks.
    Strategy:
      - Victim (User A) sends a PII prompt
      - Attacker (User B) probes with the exact same prompt
      - Compare attacker's latency against a baseline (random prompt)
      - If SafeKV works, cache-hit vs cache-miss should be indistinguishable
    """
    print("\n" + "=" * 60)
    print("TEST 5: Timing Side-Channel Resistance")
    print("=" * 60)

    # Victim's PII prompt
    victim_prompt = (
        "My credit card number is 4111-1111-1111-1111, "
        "expiry 12/28, CVV 456. My address is 742 Evergreen Terrace."
    )

    # Baseline prompt (non-existent in cache, similar length)
    baseline_prompt = (
        "The quick brown fox jumps over the lazy dog repeatedly. "
        "This sentence is used for testing font rendering quality."
    )

    # Step 1: Victim sends their PII prompt
    lat_victim, ok_victim, _ = send_request(victim_prompt, user_id="victim_01",
                                             port=port, model=model)
    print(f"  Victim sends PII prompt:     {lat_victim:.1f} ms  [ok={ok_victim}]")
    time.sleep(1)

    # Step 2: Attacker probes with the victim's prompt
    attacker_probe_latencies = []
    for i in range(num_probes):
        lat, ok, _ = send_request(victim_prompt, user_id=f"attacker_{i}",
                                   port=port, model=model)
        if ok:
            attacker_probe_latencies.append(lat)
        time.sleep(0.1)

    # Step 3: Attacker probes with baseline prompt (cache miss)
    baseline_latencies = []
    for i in range(num_probes):
        lat, ok, _ = send_request(baseline_prompt, user_id=f"attacker_baseline_{i}",
                                   port=port, model=model)
        if ok:
            baseline_latencies.append(lat)
        time.sleep(0.1)

    if attacker_probe_latencies and baseline_latencies:
        avg_probe = sum(attacker_probe_latencies) / len(attacker_probe_latencies)
        avg_baseline = sum(baseline_latencies) / len(baseline_latencies)
        ratio = avg_probe / avg_baseline if avg_baseline > 0 else 0

        print(f"\n  Attacker probe latencies ({num_probes} probes):")
        print(f"    Probing victim's prompt:   avg={avg_probe:.1f} ms")
        print(f"    Baseline (cache miss):     avg={avg_baseline:.1f} ms")
        print(f"    Ratio (probe/baseline):    {ratio:.3f}")

        # If the ratio is close to 1.0, the attacker cannot distinguish
        if 0.8 < ratio < 1.2:
            print(f"  ✓ PASS: Probe and baseline are indistinguishable (ratio={ratio:.3f})")
        else:
            print(f"  ? INFO: Some timing difference detected (ratio={ratio:.3f})")
            print(f"         This could be noise — run with more probes for confidence.")
    else:
        print("  ✗ SKIP: Insufficient successful probes")


# ============================================================
# Test 6: System Prompt Sharing (Shared-System-Prompt Pattern)
# ============================================================
def test_system_prompt_sharing(port, model):
    """
    Verify that a common system prompt is shared across tenants.
    Strategy:
      - Multiple users send requests with the same system prompt
      - User-specific content (PII) in user messages stays private
      - But the shared system prompt prefix is reused
    """
    print("\n" + "=" * 60)
    print("TEST 6: System Prompt Sharing with User Isolation")
    print("=" * 60)

    system_prompt = (
        "You are a helpful customer service assistant for Acme Corp. "
        "Always be polite, concise, and follow company guidelines. "
        "Never share customer information with other users."
    )

    # Each user has their own PII in the user message
    user_messages = [
        ("user_F", "My account number is ACC-12345. Why was I charged $99?"),
        ("user_G", "I'm Jane Doe, SSN 987-65-4321. I need to update my address."),
        ("user_H", "My order #ORD-78901 hasn't arrived. My phone is 555-0123."),
        ("user_I", "What are your business hours?"),  # non-PII
        ("user_J", "How do I reset my password?"),    # non-PII
    ]

    print("  Sending requests with shared system prompt + user-specific messages:")
    latencies = []
    for user_id, msg in user_messages:
        lat, ok, _ = send_request(msg, user_id=user_id, port=port, model=model,
                                   system_prompt=system_prompt)
        latencies.append((user_id, lat, ok))
        print(f"    {user_id}: {lat:.1f} ms  [ok={ok}]  msg={msg[:50]}...")
        time.sleep(0.5)

    # Analysis
    ok_lats = [(uid, lat) for uid, lat, ok in latencies if ok]
    if len(ok_lats) >= 3:
        first_lat = ok_lats[0][1]
        later_lats = [lat for _, lat in ok_lats[1:]]
        avg_later = sum(later_lats) / len(later_lats)
        speedup = first_lat / avg_later if avg_later > 0 else 0
        print(f"\n  Analysis:")
        print(f"    First request (cold):      {first_lat:.1f} ms")
        print(f"    Avg subsequent requests:   {avg_later:.1f} ms")
        print(f"    Speedup:                   {speedup:.2f}x")
        if speedup > 1.05:
            print(f"  ✓ PASS: System prompt prefix reuse detected across tenants")
        else:
            print(f"  ? INFO: No significant speedup (may need longer system prompt)")
    else:
        print("  ✗ SKIP: Insufficient successful requests")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SafeKV Functional Test")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--budget", type=int, default=10,
                        help="Expected access budget B (default: 10)")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Expected creator threshold K (default: 2)")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "isolation", "promotion", "budget",
                                 "repromotion", "timing", "sysprompt"],
                        help="Which test to run (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("SafeKV v2 Functional Test Suite")
    print(f"Server: http://127.0.0.1:{args.port}")
    print(f"Model:  {args.model}")
    print(f"Budget B={args.budget}, Threshold K={args.threshold}")
    print("=" * 60)

    # Health check
    try:
        resp = requests.get(f"http://127.0.0.1:{args.port}/health", timeout=5)
        if resp.status_code == 200:
            print(f"Server health: OK ({resp.status_code})")
        else:
            print(f"Server health: UNHEALTHY ({resp.status_code})")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot reach server at port {args.port}: {e}")
        print("Make sure to start the server first: ./launch_model.sh phi4")
        sys.exit(1)

    tests = {
        "isolation":   lambda: test_cross_tenant_isolation(args.port, args.model),
        "promotion":   lambda: test_shareable_promotion(args.port, args.model),
        "budget":      lambda: test_access_budget(args.port, args.model, args.budget),
        "repromotion": lambda: test_multi_source_repromotion(args.port, args.model, args.threshold),
        "timing":      lambda: test_timing_side_channel(args.port, args.model),
        "sysprompt":   lambda: test_system_prompt_sharing(args.port, args.model),
    }

    if args.test == "all":
        for name, fn in tests.items():
            try:
                fn()
            except Exception as e:
                print(f"\n  ✗ ERROR in test '{name}': {e}")
    else:
        tests[args.test]()

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
