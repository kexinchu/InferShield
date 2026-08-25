# Token-recovery ASR vs PromptPeek

Protocol (aligned, **not** the cached_tokens oracle that hit 100%):

- 50 trials × 5 slots (250 token decisions)
- Decision = argmin mean **TTFT** after pairwise LPM races
- Remote one-way RTT jitter on observed TTFT
- Open search set: true continuation missing with p=0.05 (PromptPeek-style open vocab)
- Repeats suppress noise; req/token target 48–150 (PromptPeek 148–306)
- `cached_tokens` is logged only as an oracle diagnostic

| Work / policy | Token ASR | Secret ASR | req/token | n |
|---|---|---|---|---|
| PromptPeek input / template / whole-prompt | 99% / 98% / 95% | 95% (whole) | 148–306 | paper |
| EarlyBird | 89% | — | 113 | paper |
| Phi-4 vanilla (oracle `cached_tokens`, old) | 100% | 100% | 20 | 250 — **rejected** |
| Phi-4 vanilla sequential TTFT, no LPM | 0–40% | 0% | 48–120 | smoke — channel too weak |
| Phi-4 vanilla pairwise LPM + 8ms jitter, closed set | 100% (15/15) | 100% (3/3) | 60 | smoke — too high |
| Phi-4 vanilla pairwise LPM n=50 (quarantined) | 94.0% | 76% | 60 | 250 — keep only as old-protocol note |
| Phi-4 Strict pairwise LPM n=50 (quarantined) | 94.8% | 80% | 60 | 250 — **invalid**: concurrent LPM, isolation not measured |
| **Cross-user sequential TTFT (in progress)** | TBD | TBD | ~36 | attacker ≠ victim, fresh principal per probe |

Why the old run hit 100% while PromptPeek is 95%: same-host API, closed set, and `cached_tokens` as the decision bit (no remote noise). The campaign removes that oracle.
