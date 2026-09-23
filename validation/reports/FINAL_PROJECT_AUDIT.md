# FINAL PROJECT AUDIT

## Status

**PROJECT STATUS: COMPLETE AVAILABILITY-CONSTRAINED RESEARCH CYCLE**  
**ENGINEERING STATUS: PASS**  
**RESEARCH VALIDATION: LIMITED**  
**PAPER READY: YES, WITH LIMITATIONS**

The final RUN 6 experiment was completed only for actions that passed real provider validation. No unavailable provider was fabricated, and the experimental policy was not promoted to production.

## Action Availability

Full configured action space: `A0-A6`. `A7` is permanently masked.

Final executable action space:

- A0 `gemma-3-4b`
- A1 `qwen-coder-3b`
- A2 `deepseek-r1-7b`
- A6 `meta-llama/llama-3.3-70b-instruct`

Excluded from contemporaneous RUN 6 evaluation:

- A3 Gemini: `429 RESOURCE_EXHAUSTED` quota
- A4 Mistral: `429` rate limit
- A5 Groq: `403 Forbidden` authorization failure
- A7 BGE-M3: embedding model, permanently masked

Direct Ollama validation confirmed A0, A1, and A2. A6 passed a direct OpenRouter execution probe.

## RUN 6 Dataset

Dataset: `data/evaluation/run6/final_available_actions_direct/run6_available_actions.jsonl`  
SHA-256: `f479826d8c96f62504c62c9ba8f08a68ccae4433602a50badd278a541deb336f`

| Metric | Result |
|---|---:|
| Raw records | 72 |
| Valid records | 72 |
| Duplicates | 0 |
| A0 | 18 |
| A1 | 18 |
| A2 | 18 |
| A3 | 0 |
| A4 | 0 |
| A5 | 0 |
| A6 | 18 |
| A7 | 0 |
| Success rate | 100% |
| Fallback rate | 0% |
| Reward mean / median | 0.9332 / 0.9700 |
| Latency mean / median / P95 | 12119.76 / 5686.21 / 31924.60 ms |
| Propensity min / max / mean | 1.0 / 1.0 / 1.0 |
| Measured cost | `$0.000772` |

Selected and executed actions matched exactly for all 72 records. No A7 generation occurred.

## Training

The existing `PolicyTrainer` trained only on the isolated RUN 6 training split:

- train samples: 50
- held-out test samples: 22
- K: 8
- D: 12
- epochs: 100
- learning rate: 0.01
- L2 regularization: 0.01
- seed: 42
- training MSE: 0.0059

Experimental artifact: `data/rl/models/rl_contextual_bandit_policy_run6_available_actions.json`  
Artifact SHA-256: `621cedee51863063a29cdb56ff8860a37039934052ebb04cb8d2359df7b616ad`

Production artifact: `data/rl/models/rl_contextual_bandit_policy.json`  
Production SHA-256: `f0e8f61370931eb9cca7421d0ac8a016b46eac7d31c87622cfe9e7d2b2f1fc71`  
Production promotion: **NO**.

## Availability-Constrained Evaluation

Regime: **Availability-Constrained RUN 6 Evaluation**. Evaluation used the held-out 22 records from the same executable-action regime.

- policy agreement: `0/22` (`0.0%`)
- IPS: `0.0`
- SNIPS: `0.0`
- ESS: `0.0`
- positivity coverage: `0.0` for held-out deterministic behavior/policy overlap
- observed held-out reward mean: `0.8981`
- observed held-out reward median: `0.9343`
- held-out success rate: `100%`
- held-out latency mean: `9542.96 ms`
- held-out cost: `$0.000772`

No paired online RL rollout was performed. IPS/SNIPS/ESS evidence is weak and insufficient for policy superiority claims. Training MSE is not treated as generalization performance. No RL-vs-baseline superiority claim is made.

## Engineering Validation

- backend suite: `247 passed, 1 skipped, 0 failed`
- runtime probe: passed RNG persistence, gating, epsilon behavior, propensity, selected/executed separation, and A7 masking
- compileall: passed
- frontend build: passed, 1560 modules transformed
- production artifact hash: unchanged through final training/evaluation

The production artifact and RUN 1-RUN 5 artifacts were not overwritten. The experimental artifact has a provenance manifest at [validation/results/run6/run6_available_actions_manifest.json](../results/run6/run6_available_actions_manifest.json).

## Historical Runs and Limitations

- RUN 2: controlled three-local-model comparison
- RUN 3: complex-task orchestration experiment
- RUN 4: synthesis optimization experiment
- RUN 5: preliminary RL training/evaluation with insufficient ESS and coverage
- RUN 6: availability-constrained contextual-bandit training and offline evaluation

Third-party provider quota/authorization constraints prevented complete contemporaneous evaluation across every configured cloud action. These results are not evidence across the complete seven-generation-model action space.

Pytest now redirects no-path test buffers to a temporary session path and resets the singleton between tests. The canonical buffer remained unchanged across the full suite: 28 records and SHA-256 `745d90d86568164381bc4e361f91e4b66023760e6859be6c60fafbc69c08fb0d` before and after. The production no-path service still resolves to `data/rl/experience_buffer.jsonl` when the pytest-only override is absent.

## Final Gate

**RUN 6 STATUS: COMPLETE — AVAILABILITY-CONSTRAINED**  
**PRODUCTION PROMOTION: NO**  
**PAPER READY: YES, WITH LIMITATIONS**

RL superiority remains **not established**. The full A0-A6 action-space evaluation was not possible because A3/A4/A5 were unavailable; RUN 6 evaluated A0/A1/A2/A6 only. The held-out RL policy had zero policy/behavior overlap, so IPS/SNIPS/ESS were zero. This means the logged data did not provide usable off-policy overlap; it does not prove the RL algorithm failed. The experimental policy was not promoted, and production remains protected by the existing artifact and baseline fallback.
