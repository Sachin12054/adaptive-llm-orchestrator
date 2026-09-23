# AUDIT 37 — Final Runtime Validation

**Date:** 2026-09-21  
**Scope:** Runtime validation only. RUN 6 data collection was not started. No RL training was run. No trained artifact or RUN_1–RUN_5 result was modified.

## Final Gate

**READY FOR RUN 6 DATA COLLECTION: NO — AUTHORITATIVE PRODUCTION ARTIFACT UNRESOLVED**

The three requested engineering blockers are repaired and validated. The existing production artifact was not restored or guessed because its authoritative provenance cannot be established from the current repository state.

## 1. Singleton Contamination Repair

The original blocker was a custom temporary `persistence_path` rebinding the global `ExperienceBufferService` singleton. The repair keeps the no-argument production service pinned to:

`data/rl/experience_buffer.jsonl`

Custom persistence paths now create isolated instances. Regression coverage verifies canonical path restoration and temp-buffer isolation.

## 2. Test Evidence

| Suite | Result |
|---|---:|
| RL production migration | 10 passed |
| Cost, complex-task, and S2 regressions | 17 passed |
| RL experience collection and adaptive decision | 21 passed |
| Full backend suite | 247 passed, 1 skipped, 0 failed |

The full command was:

```text
PYTHONPATH=backend ./venv/Scripts/python.exe -m pytest backend/tests -q
```

The full suite completed with `247 passed, 1 skipped, 659 warnings`.

## 3. RNG Persistence and Reproducibility

**Result: PASS.**

Runtime probe evidence:

- `EXPLORATION_SEED = 42`
- `EXPLORATION_SEED = 42`
- Same-engine sequence at `epsilon=1.0`: `['gemma-3-4b', 'gemma-3-4b', 'deepseek-r1-7b', 'deepseek-r1-7b', 'gemma-3-4b']`
- Fresh-engine sequence: identical
- Same-engine RNG state advances: `True`
- Identical-seed reproducibility: `True`
- RNG access is protected by the engine lock.

## 4. Exploration Gating

**Result: PASS for gating behavior.**

Runtime observations:

- `RL_DATA_COLLECTION_MODE=False`: policy `rl_contextual_bandit_policy`, deterministic selected action, propensity `1.0`, no exploration suffix.
- `RL_DATA_COLLECTION_MODE=True`: policy `rl_contextual_bandit_policy_exploration`, candidate probabilities populated, and configured epsilon applied.
- `BAAI/bge-m3` remained at probability `0.0` in both modes.

## 5. Epsilon-Greedy Validation

**Result: PASS for available runtime candidates.**

The live local registry exposed three configured valid generation candidates: Gemma, Qwen, and DeepSeek, so runtime `N=3`. Rounded probability sums were within `1e-5` of `1.0`.

| Mode | Epsilon | Selected | Probability sum | Selected propensity | A7 probability |
|---|---:|---|---:|---:|---:|
| off | 0.0 | gemma-3-4b | 1.00000 | 1.00000 | 0.0 |
| on | 0.0 | gemma-3-4b | 1.00000 | 1.00000 | 0.0 |
| on | 0.1 | gemma-3-4b | 0.99999 | 0.93333 | 0.0 |
| on | 0.5 | gemma-3-4b | 1.00001 | 0.66667 | 0.0 |
| on | 1.0 | gemma-3-4b | 0.99999 | 0.33333 | 0.0 |

The measured probabilities match:

- greedy: $(1-\epsilon)+\epsilon/N$
- non-greedy: $\epsilon/N$

The implementation rounds each probability to five decimals, accounting for the small normalization residual.

## 6. Action 7 Masking

**Result: PASS.**

Runtime candidate probability for `A7 = BAAI/bge-m3` was `0.0` for epsilon values `0.0`, `0.1`, `0.5`, and `1.0`. The policy masks it from generation candidates while the embedding configuration remains `BAAI/bge-m3`.

## 7. Selected Versus Executed Action and Propensity

**Result: PASS.**

Controlled fallback evidence:

- RL-selected action: `A3`, `gemini-3.5-flash`
- Executed action: `A0`, `gemma-3-4b`
- `fallback_used`: `true`
- `fallback_reason`: `rate_limit`
- `propensity_probability`: `0.82857`

The stored propensity matched the selected action's behavior-policy probability, not the fallback execution action. The existing runtime probe also verified this separation.

## 8. Real Experience Record

**Result: PASS.**

A real local orchestration request produced a persisted record containing:

- 12-dimensional `state`
- `rl_selected_action` and `rl_selected_model`
- `executed_action` and `executed_model`
- `propensity_probability` and candidate probability map
- reward and terminal state
- timestamp
- provider and execution status
- pipeline latency
- fallback fields
- cost, currency, and cost source

Measured local record values included `executed_provider=ollama`, `cost=0.0`, `cost_source=zero_local`, `fallback_used=false`, and `propensity_probability=1.0`.

## 9. Action Registry

**Result: PASS.**

The runtime registry was:

| Action | Model | Generation |
|---|---|---|
| A0 | gemma-3-4b | allowed |
| A1 | qwen-coder-3b | allowed |
| A2 | deepseek-r1-7b | allowed |
| A3 | gemini-3.5-flash | allowed |
| A4 | mistral-small-latest | allowed |
| A5 | llama-3.3-70b-versatile | allowed |
| A6 | meta-llama/llama-3.3-70b-instruct | allowed |
| A7 | BAAI/bge-m3 | masked |

The same authoritative `ACTION_MAP`/`REVERSE_ACTION_MAP` is used by the experience buffer, RL policy, and decision engine. Telemetry records action IDs and canonical model IDs. The frontend displays the RL production/fallback telemetry but does not define a conflicting action map.

## 10. Production RL Routing

**Result: PASS.**

Runtime configuration:

- `PRODUCTION_POLICY=rl`
- `FALLBACK_POLICY=baseline`
- primary policy: `RLContextualBanditPolicy`
- safety policy: `BaselineAdaptivePolicy`

Normal runtime decisions reported `policy=rl_contextual_bandit_policy` and `production_policy=rl_contextual_bandit_policy`; fallback telemetry was false when the RL decision was valid. No shadow-only override was observed.

## 11. No Online Training

**Result: PASS.**

After production decisions:

- `W` shape: `(8, 12)` and unchanged
- `b` shape: `(8,)` and unchanged
- policy version: `2.0.0` and unchanged
- no training call was performed

## 12. Trained Artifact Integrity

**Result: PASS for mutation protection; provenance unresolved.**


The training API regression now injects a temporary `PolicyTrainer(output_path=tmp_path/...)`. The temporary artifact is created and the production hash remains unchanged. The current production artifact is not Git-tracked, while the archived RUN 5 artifact has different provenance. **AUTHORITATIVE PRODUCTION ARTIFACT UNRESOLVED.** No weights were fabricated or overwritten.

### Final Provenance Check

| Candidate | SHA-256 | Metadata / evidence |
|---|---|---|
| `data/rl/models/rl_contextual_bandit_policy.json` | `f0e8f61370931eb9cca7421d0ac8a016b46eac7d31c87622cfe9e7d2b2f1fc71` | v2.0.0; 38 samples; state 12; weights `(8,12)`; bias `(8,)`; untracked; timestamp `1789844807.429648` |
| `validation/results/run5/run5_trained_policy.json` | `580287c43b0de7bd97b46046eb010ec99d960f04e1a769299c21e3d5978446da` | v2.0.0; 14 samples; state 12; weights `(8,12)`; bias `(8,)`; RUN 5 metadata: 100 epochs, lr 0.01, L2 0.01, seed 42 |
| `data/rl/models/rl_contextual_bandit_policy_offline_v2.json` | `2dcc4e0babc890961c6953fa71dc88c5d1a2ee9703ac30f579fde884c7de0775` | v2.0.0-offline; 87 samples; state 12; weights `(8,12)`; bias `(8,)` |
| `data/rl/models/rl_contextual_bandit_policy_offline_v3.json` | `c9bdfea8dd8a831a78c6a3a995cfbb54a9f47a6a9016187bae1e0974317acebf` | v3.0.0-propensity-aware; 70 samples; state 12; weights `(8,12)`; bias `(8,)` |
| `data/rl/models/rl_contextual_bandit_policy_offline_v4.json` | `f6282856f72a1ac1c08f21327f5f3df6621165f748ccd4048451c1bd5d77867f` | v4.0.0-final-validated; 70 samples; state 12; weights `(8,12)`; bias `(8,)` |

Evidence conclusion:

- Git history contains no tracked production artifact or RUN 5 artifact entry; `git ls-files` and path history returned no record for either file.
- RUN 5's manifest explicitly identifies `580287...` as the RUN 5 artifact hash and `41f94...` as the production hash at manifest creation. `41f94...` matches none of the current candidate files.
- Current production weights and bias are numerically unequal to RUN 5 and every archived offline candidate.
- The current production file reports 38 training samples, while the current buffer contains 28 valid records; therefore it cannot be reproduced from the present buffer snapshot. This is consistent with a prior training/test output, but no preserved 38-record dataset or manifest links it authoritatively to RUN 5.
- Version and tensor shapes match multiple candidates and are insufficient provenance evidence.

Classification: the current file cannot be confidently identified as the authoritative RUN 5 artifact or as a documented derivative. It is most consistent with an unproven prior training/test-generated artifact, but that classification cannot be promoted to confirmed provenance.

## 13. Fallback Validation

**Result: PASS.**

- Corrupt artifact with `state_dim=99`: baseline fallback activated; policy was `baseline_adaptive_policy`; `fallback_used=true`.
- Missing artifact: RL weights remain unavailable; baseline policy selected with `fallback_used=true` and a populated fallback reason.
- Invalid dimensions and invalid version: both rejected and routed to baseline.

## 14. Cost Tracking

**Result: PASS.**

Runtime calculator evidence:

- local Ollama: `cost=0.0`, `cost_source=zero_local`, currency `USD`
- cloud Gemini: `cost=0.00045`, `cost_source=configured_pricing_estimate`

The cloud value is a configured pricing estimate, not a billing claim. S2 deterministic assembly returned cost `0.0` on both executions.

## 15. Complex Task / S2 Regression

**Result: PASS for focused regression.**

Focused complex-task and synthesis tests passed. Direct S2 runtime execution was byte-for-byte reproducible, returned `0.0` synthesis cost, and did not include `BAAI/bge-m3` as a generation model. The full complex-task regression suite also passed.

## 16. RUN 5 Historical Discrepancy

**Result: UNDERSTOOD; historical artifacts unchanged.**

The archived RUN 5 report states:

- training samples: `14`
- action coverage: Gemma `20`, Qwen `1`

The runner explains the difference:

1. It reads and validates the full experience buffer.
2. It sorts valid records chronologically.
3. It takes a 70% temporal split for training: `int(N_valid * 0.7)`, with a minimum of one.
4. The action coverage table uses `action_counts` accumulated over **all valid deduplicated buffer records**, not the temporal training subset.

Therefore `14` is the training split count, while `20` and `1` are full-buffer action-support counts. They are different denominators, not a silent reconciliation or fabricated data. The archived `validation/results/run5/train_manifest.json` independently records `record_count: 14` and the same training-ID hash used by the report.

The current live buffer has since accumulated a different record set and was not used to rewrite RUN 5 artifacts.

## 17. Final Check Table

| CHECK | RESULT | EVIDENCE |
|---|---|---|
| Full backend suite | PASS | 247 passed, 1 skipped, 0 failed |
| RNG persistence/reproducibility | PASS | Same-engine state advanced; fresh engine reproduced the seed-42 sequence |
| Exploration gating | PASS | Mode off deterministic; mode on exploration policy/probabilities |
| Epsilon mathematics | PASS | Runtime N=3 probabilities match formulas; sums within 1e-5 |
| Probability normalization | PASS | Sums 0.99999–1.00001 |
| Propensity correctness | PASS | Selected/fallback case retained selected-action probability 0.82857 |
| Action 7 masking | PASS | A7 probability 0.0 at all tested epsilons |
| Selected/executed distinction | PASS | A3 selected, A0 executed, fallback metadata preserved |
| Experience schema | PASS | Real orchestration record contained state, actions, reward, latency, status, cost, fallback, timestamp |
| Canonical action registry | PASS | A0–A7 matched across runtime registry and policy |
| RL production routing | PASS | `PRODUCTION_POLICY=rl`, RL primary, baseline safety policy |
| Baseline fallback | PASS | Missing, corrupt, invalid-dimension, and invalid-version artifacts route to baseline |
| No online parameter updates | PASS | W, b, and policy version unchanged |
| Artifact integrity | PASS for mutation protection | Hash unchanged before/after final suite: `f0e8f6...1fc71` |
| Cost tracking | PASS | local `zero_local/$0.00`; cloud `configured_pricing_estimate` |
| Complex/S2 regression | PASS | Focused tests pass; S2 deterministic and $0.00 |
| RUN 5 discrepancy | PASS | 14 is temporal train split; 20/1 are full valid-buffer action counts |

## Final Status

**READY FOR RUN 6 DATA COLLECTION: NO — AUTHORITATIVE PRODUCTION ARTIFACT UNRESOLVED**

The three requested blocker repairs pass. Do not start RUN 6 until the authoritative production artifact provenance is confirmed or explicitly accepted by the project owner; no artifact restoration was guessed.
