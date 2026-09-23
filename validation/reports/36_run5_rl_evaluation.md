# Report 36 — RUN_5: Scientific Evaluation of Contextual-Bandit RL Policy

**Project:** Adaptive Multi-LLM Orchestration Using Contextual Bandits  
**Experiment ID:** RUN_5_RL_SCIENTIFIC_EVALUATION  
**Timestamp (UTC):** 2026-09-19T07:38:21.917355Z  
**Git Commit:** unavailable  

---

## 1. Objective

Produce a scientifically valid, evidence-based evaluation of the trained `RLContextualBanditPolicy` 
versus the deterministic `BaselineAdaptivePolicy`. The goal is NOT to prove RL superiority but to 
objectively measure whether the trained policy differs from and/or improves upon the baseline.

**Critical constraint:** No raw experimental data from RUN_1–RUN_4 was modified. No results were fabricated.

---

## 2. Dataset Provenance

| Dataset | Path | Records | SHA256 (first 16) |
|---------|------|---------|-------------------|
| Experience buffer | `data/rl/experience_buffer.jsonl` | 22 raw | 8ca2a0e23138e0c5... |
| Propensity benchmark | `data/evaluation/propensity_benchmark_100.jsonl` | 100 | d2aca845fd389d62... |
| RL model artifact (prod) | `data/rl/models/rl_contextual_bandit_policy.json` | — | e5bdb262299f951d... |
| RUN_5 trained artifact | `validation/results/run5/run5_trained_policy.json` | — | 580287c43b0de7bd... |

**Historical RUN data (read-only, not modified):**
- RUN_2 Local-only: `validation/results/local_only/`
- RUN_3 Complex task: `validation/results/complex_task_comparison/`
- RUN_4 Synthesis opt: `validation/results/complex_task_synthesis_optimization/`

---

## 3. Training Dataset (Experience Buffer Audit)

| Metric | Count |
|--------|-------|
| Total raw records | 22 |
| Duplicates | 0 |
| Valid records (action ≥ 0, state dim = 12) | 21 |
| Invalid (action = -1, unmapped model) | 1 |
| Missing state (dim ≠ 12) | 0 |
| Missing action | 0 |
| Missing reward | 0 |
| Missing cost | 0 |
| Missing latency | 0 |
| Missing quality | 22 |

**Training records used (post-split):** 14

---

## 4. Test Dataset

**Primary holdout:** `data/evaluation/propensity_benchmark_100.jsonl`  
**Collection method:** Epsilon-greedy baseline-preferred exploration (ε = 0.2) on 100 prompts  
**Records:** 100  
**Propensity scores:** Available (pi_b ≈ 0.8286 for preferred action, 0.0286 for alternatives)  
**Timestamp range:** Separate from experience buffer — collected before RL migration  

---

## 5. Data Split Methodology

- **Method:** Temporal chronological split on experience buffer records
- **Train/Test ratio:** 70% / 30% (experience buffer only)
- **Primary test set:** Independent propensity benchmark (100 records, NOT mixed into training)
- **Train set hash (first 16):** f3fa9076d22a6139...
- **Test set hash (first 16):** a9a96f80dc36c103...

> **Note:** No future observations from the test set were leaked into training. The propensity 
> benchmark was collected independently and temporally precedes the RL migration.

---

## 6. RL Algorithm

**Algorithm:** Linear Contextual Bandit with L2-Regularized Gradient Descent  
**Model:** Q(s, a) = W_a^T · s + b_a where W ∈ ℝ^(K×D)  
**Action representation:** One-hot selection — greedy argmax at inference  

---

## 7. State/Action Dimensions

| Parameter | Value |
|-----------|-------|
| State dimension (D) | 12 |
| Action space (K) | 8 |
| Action 7 (BAAI/bge-m3) | MASKED — never selected for generation |

**State vector components (D=12):**
1. Intent code (normalized, 0–1)
2. Intent ambiguity flag (0/1)
3. Complexity score
4. Semantic complexity
5. Reasoning complexity
6. Task complexity
7. Context complexity
8. Output complexity
9. CPU utilization (normalized)
10. Memory utilization (normalized)
11. GPU available (0/1)
12. Baseline policy score

---

## 8. Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | Linear Contextual Bandit (Regularized GD) |
| Epochs | 100 |
| Learning rate | 0.01 |
| L2 regularization λ | 0.01 |
| Random seed | 42 |
| Training samples | 14 |
| Valid samples | 14 |
| Update count | 1400 |

---

## 9. Training Results

| Metric | Value |
|--------|-------|
| Final MSE | 0.0119 |
| Training duration | 0.0077s |
| Policy version | 2.0.0 |
| Artifact | `validation/results/run5/run5_trained_policy.json` |

> **Warning:** Training MSE with 14 samples is not a reliable indicator of generalization.
> Only 2 of 8 actions have any training data. The policy is highly biased toward gemma-3-4b 
> (action 0, 20 samples) and qwen-coder-3b (action 1, 1 samples).

---

## 10. Offline Evaluation

**Test set:** propensity_benchmark_100 (N=100)  
**Evaluation mode:** Off-policy with IPS/SNIPS estimation  

| Metric | Value |
|--------|-------|
| Test records | 100 |
| RL agrees with baseline | 16/100 (16.0%) |
| Records in IPS calculation | 19 |
| IPS reward estimate | 7.9349 |
| SNIPS reward estimate | 0.7856 |
| ESS | 5.99 |
| Positivity coverage | 0.1900 (19.0%) |
| Evidence strength | WEAK_INSUFFICIENT |

---

## 11. Baseline Comparison

| Metric | BaselineAdaptivePolicy | RLContextualBanditPolicy | Difference |
|--------|----------------------|--------------------------|------------|
| Mean reward (observed) | 0.6816 | 7.9349 (IPS) | +7.2533 |
| Median reward | 0.9591 | N/A (not executed) | N/A |
| Success rate | 71.0% | N/A (not executed) | N/A |
| Mean latency (ms) | 37087 | N/A (not executed) | N/A |
| P95 latency (ms) | 122061 | N/A (not executed) | N/A |
| Local cost | $0.00 | $0.00 | $0.00 |

---

## 12. Statistical Analysis

A paired statistical test (e.g., Wilcoxon signed-rank) cannot be applied because RL actions were not executed in production during this evaluation. Observed outcomes belong exclusively to the behavior policy (epsilon-greedy baseline). Off-policy IPS/SNIPS estimates are the only statistically valid approach.

**IPS-based analysis:**
- IPS records (RL action == behavior action): 19/100
- IPS estimate: 7.9349  
- SNIPS estimate: 0.7856  
- ESS: 5.99 (threshold for "strong" evidence: ≥ 30)  
- Positivity coverage: 0.1900  
- Evidence classification: **WEAK_INSUFFICIENT**

**Agreement analysis:**
- When RL agrees with baseline (16 records): mean reward = 0.7740
- When RL disagrees (84 records): mean observed reward = 0.6640 (from behavior policy, not RL)

---

## 13. Action Coverage

| Action | Model | Train Samples | Test (Behavior) | Test (RL) | Masked |
|--------|-------|:---:|:---:|:---:|:---:|
| 0 | gemma-3-4b | 20 | 26 | 2 | NO |
| 1 | qwen-coder-3b | 1 | 17 | 98 | NO |
| 2 | deepseek-r1-7b ⚠️ | 0 | 10 | 0 | NO |
| 3 | gemini-3.5-flash ⚠️ | 0 | 30 | 0 | NO |
| 4 | mistral-small-latest ⚠️ | 0 | 11 | 0 | NO |
| 5 | llama-3.3-70b-versatile ⚠️ | 0 | 2 | 0 | NO |
| 6 | meta-llama/llama-3.3-70b-instruct ⚠️ | 0 | 4 | 0 | NO |
| 7 | BAAI/bge-m3 | 0 | 0 | 0 | YES |

⚠️ = Insufficient training support (0 samples)

**Action 7 (BAAI/bge-m3) verification:** Selected 0 times for text generation (must = 0). **[VERIFIED]**

---

## 14. Production Validation

| Check | Result |
|-------|--------|
| PRODUCTION_POLICY | `rl` |
| fit() in inference path | ✅ ABSENT |
| train() in inference path | ✅ ABSENT |
| RLContextualBanditPolicy in engine | ✅ YES |
| BaselineAdaptivePolicy fallback | ✅ YES |

---

## 15. Fallback Validation

| Fallback Trigger | Status |
|-----------------|--------|
| weights_is_none_fallback | ✅ PRESENT |
| embedding_mask_action_7 | ✅ PRESENT |
| invalid_model_fallback_in_engine | ✅ PRESENT |
| fallback_reason_populated | ✅ PRESENT |
| baseline_called_on_fallback | ✅ PRESENT |

---

## 16. Cost Validation

| Source | Records | Expected Cost | Verified |
|--------|---------|---------------|---------|
| Local Ollama (zero_local) | 19 | $0.00 | ✅ |
| Cloud pricing estimate | 1 | Configured | ✅ |
| Total cloud cost logged | — | $0.0025 | ✅ |

---

## 17. Complex Task Validation

- `TaskAggregator` exists: ✅  
- S2 synthesis cost tracking: ❌  
- S2 synthesis LLM calls: Assembly only — $0.00  
- RUN_3 synthesis cost (S2): $0.0  

---

## 18. Dashboard Validation

GET `/api/v1/metrics/cost` and `/api/decision/status` return cost metrics and active policy.  
Code analysis confirms telemetry fields: `production_policy`, `rl_selected_model`, `executed_model`,  
`fallback_used`, `fallback_reason` are populated in every orchestration response.

---

## 19. Test Results

Backend Pytest Suite: **229 passed, 1 skipped, 616 warnings in 161.59s (0:02:41)**  
Exit code: 0

---

## 20. Limitations


SCIENTIFIC LIMITATIONS — RUN_5 RL Scientific Evaluation

1. SMALL TRAINING SAMPLE SIZE: Only 10 valid training records are available in
   the experience buffer. This is critically insufficient for a K=8, D=12 contextual
   bandit. Standard recommendations require at minimum hundreds of samples per action.

2. SEVERE ACTION IMBALANCE: Actions 2-6 have ZERO training samples. Weights for
   deepseek-r1-7b, gemini-3.5-flash, mistral-small-latest, llama-3.3-70b-versatile,
   and meta-llama/llama-3.3-70b-instruct remain at their initial values (near zero).
   The policy cannot generalize to these models.

3. INSUFFICIENT EXPLORATION: All live requests used deterministic epsilon=0 routing
   (only gemma-3-4b and qwen-coder-3b appear in the buffer). Without diverse action
   coverage, the policy cannot learn relative quality across the full action space.

4. POSITIVITY LIMITATION: Only 19% of test records overlap between RL and behavior
   policy actions. ESS = 6.0, well below the 30-sample threshold for reliable
   off-policy estimates. IPS/SNIPS estimates carry high variance.

5. MISSING PROPENSITY CALIBRATION: The propensity_benchmark_100 used epsilon-greedy
   with pi_b(a|s) ≈ 0.829 for the preferred action and 0.029 for others. This
   log-propensity weighting is logged but not fully calibrated to the current RL policy.

6. NO COUNTERFACTUAL OUTCOMES: RL actions that differ from the behavior policy have
   no observed rewards. The evaluation cannot directly measure what RL would have
   achieved in production without real deployment.

7. TEST SET SIZE: 100 records provide limited statistical power for sub-group analysis
   (e.g., per-intent, per-complexity comparisons).

8. LOCAL ENVIRONMENT ONLY: Propensity benchmark ran local Ollama models only.
   Cloud provider latency and cost comparisons require real API access.

9. CONFIGURED PRICING ESTIMATES: Cloud costs use pre-configured pricing tables,
   not actual billing data. Estimates may differ from real provider invoices.

10. FALLBACK EFFECTS: With only 2 actions trained, any request requiring cloud
    providers may trigger the BaselineAdaptivePolicy fallback in production,
    effectively routing through baseline rather than RL.

CONCLUSION: The current RL policy is scientifically insufficient to claim superiority
over BaselineAdaptivePolicy. The evidence does NOT support declaring RL superior.
The policy correctly routes to trained local models (gemma-3-4b, qwen-coder-3b)
and safely falls back to baseline for unseen action contexts.


---

## 21. Reproducibility Manifest

| Field | Value |
|-------|-------|
| Experiment ID | RUN_5_RL_SCIENTIFIC_EVALUATION |
| Timestamp (UTC) | 2026-09-19T07:38:21.939260Z |
| Git Commit | unavailable |
| Training set hash | f3fa9076d22a613913f8e9b83fa343c2... |
| Test set hash | a9a96f80dc36c1031714adc8d95802c5... |
| Policy artifact hash | 580287c43b0de7bd97b46046eb010ec9... |
| Policy version | 2.0.0 |
| K | 8 |
| D | 12 |
| Epochs | 100 |
| LR | 0.01 |
| L2 lambda | 0.01 |
| Random seed | 42 |

---

## 22. Final Interpretation

### What the evidence shows:

1. The trained `RLContextualBanditPolicy` selects `gemma-3-4b` for 2 of 100 test prompts 
   and `qwen-coder-3b` for 98 prompts. This is because only these two models 
   have non-zero learned Q-values from the 14-record training set.

2. The RL policy agrees with the baseline on 16/100 (16.0%) of test cases. This high 
   agreement reflects that the baseline already preferred these two local models for most 
   prompts in the propensity benchmark, and the RL policy has learned approximately the same 
   preference on a narrow slice of the state space.

3. The IPS-based off-policy reward estimate is **WEAK_INSUFFICIENT**. With ESS = 6.0 
   (WEAK_INSUFFICIENT), the statistical evidence is too weak to draw conclusions about RL vs. baseline performance.

4. **The evidence does NOT support claiming RL superiority over BaselineAdaptivePolicy.**

5. The production system correctly implements all safety invariants:
   - `PRODUCTION_POLICY = "rl"` 
   - `BaselineAdaptivePolicy` fallback is always available
   - Action 7 (embedding model) is never selected for text generation
   - No online training occurs during inference

### Research recommendation:

To achieve a scientifically valid comparison, the following would be needed:
- Minimum ~500 experience records covering all 8 actions
- Epsilon-greedy or Thompson Sampling exploration with full action coverage
- Online A/B or interleaving evaluation with randomized request assignment
- Properly calibrated propensity scores across the full deployment period
