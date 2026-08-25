# SafeKV Experiment #1 analysis

## Dataset completeness

- Models: Phi-4-14B, Qwen3-30B-A3B, Qwen3-32B.
- Matrix per model: 4 attacker counts × 5 authorization conditions × 2 schedules × 20 trials.
- Raw rows: 2,400 expected, 2,400 present.
- Aggregate cells: 120 expected, 120 present.
- Failed or missing rows: 0.
- Passing rows: 2,400/2,400.
- Prefix lengths: 19–95 tokens; median 48.5 tokens.

## Acceptance results

Across all models and trials:

- Unauthorized Public promotions: 0.
- Victim-node relabels: 0.
- Private-address aliases: 0.
- Cross-tenant Private hits: 0.
- Public objects aliasing victim-private KV: 0.
- Attacker hits through a victim Private variant: 0.

For each of `none`, `forged`, `stale`, and `revoked`, all 480 trials
created zero Public objects. The registry returned the expected stable rejection
reason (`invalid_mac`, `stale_epoch`, or `revoked`) where applicable.

For `valid`, all 480 trials created exactly one separate Public object, retained
the victim-owned Private variant, and allowed the positive-control attacker
request to hit the Public object. No Public object reused victim-private KV.

Therefore all go/no-go predicates in `Exp-placeholder.md` are satisfied.

## Paper-facing result

> Across Phi-4-14B, Qwen3-30B-A3B, and Qwen3-32B, and across
> \(A\in\{1,2,4,8\}\), missing/forged/stale/revoked authorizations, and both
> sequential and concurrent schedules, SafeKV recorded zero unauthorized Public
> promotions, zero victim-node relabels, zero private-address aliases, and zero
> cross-tenant Private hits in 2,400 trials. In every valid-authorization trial,
> SafeKV materialized exactly one separate Public object, preserved the
> victim-private variant, and served the positive-control reuse only from the
> Public namespace.

## What this establishes

The measurements are deterministic implementation evidence for the Experiment
#1 contract:

1. Untrusted equal-prefix submissions did not create `Verified-Public`.
2. Private cache variants remained separated by principal.
3. Missing, forged, stale, and revoked authorizations failed closed.
4. Valid authorization remained functional and created a distinct Public
   object rather than relabeling or aliasing victim state.

## Proof and claim boundary

The data satisfy the placeholder, but experiments do not constitute a universal
proof of P2. The formal argument must state and discharge the following
assumptions, and the production implementation still has corresponding gaps:

1. **Authenticated principal binding.** The current OpenAI and native generate
   paths accept `user_id` from request-controlled sampling parameters. Without
   an authenticated gateway binding, an attacker can claim the victim's
   principal and bypass the namespace theorem's premise.
2. **Control-plane separation.** A valid signed authorization is accepted and
   materialized through the data-plane request path. HMAC provenance prevents
   forgery, but a separate protected registry/materialization API is still
   needed to match the paper's trusted-control-plane wording.
3. **Atomic insertion argument.** SGLang's single scheduler serializes the
   tested client races, but the radix insertion code does not yet use the
   declared per-variant transition locks. A proof covering genuinely parallel
   cache mutation needs explicit locking or a documented single-writer
   invariant.
4. **Scope.** This experiment validates Strict-mode Promotion Integrity and
   Private isolation only. It does not establish the Balanced-mode durable
   exposure ledger, eviction/recovery accounting, Public prewarming
   independence, timing advantage, or end-to-end identity authentication.

Accordingly, the revised Experiment #1 result is **go**, while a broader claim
that the deployed implementation fully proves all revised SafeKV security
properties is **not yet justified**.
