# Submission-gap experiments (independent of paper files)

These runs produce evidence for reviewer-facing gaps. **Do not paste into the paper until reviewed.**

| ID | Question | Output |
|----|----------|--------|
| E1 | Vanilla SGLang membership AUC under the corrected P3 protocol | `e1_vanilla_attack/` |
| E2 | Revised B∈{0,1,10,50,100} security–cost sweep | `e2_budget_sweep/` |
| E3 | Repeated serving (≥3 seeds) + Shared System Prompt emulation | `e3_serving_repeated/` |
| E4 | Authenticated principal binding (neg/pos control) | `e4_principal_binding/` |
| E5 | Stronger recovery (guaranteed / logit_topk) at B=100 | `e5_strong_recovery/` |

Orchestrator: `../run_submission_gap_all.sh`  
Aggregator: `../aggregate_submission_gap_experiments.py` → `aggregated/`

Notes:
- CacheSolidarity is unavailable and is not fabricated.
- `cache_partition` is recorded only as an alias of `strict` in E3.
- Balanced promotion now sets `BUDGETED_SHARED` visibility; E2/E5 use `--post-victim-settle-ms 750`.
