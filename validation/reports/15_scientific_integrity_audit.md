# Scientific Integrity Audit Report (Phase 9 Audit)

**Research Title**: Adaptive Multi-LLM Orchestration Using Contextual Bandits  
**Audit Timestamp**: 2026-09-03T12:18:00+05:30  
**Audit Scope**: Complete inspection of `validation/` dataset, runners, raw JSONL telemetry, quality evaluators, statistical calculations, and reports 01–14.  

---

## A. EXECUTIVE VERDICT

> [!CAUTION]
> **AUDIT CLASSIFICATION: INVALID / SYNTHETIC EXECUTION LATENCY & TEXT GENERATION**
>
> The scientific audit discovered that while the **routing decision logic** of `BaselineAdaptivePolicy` and baseline policies executed correctly, the actual **model text generation** and **execution latencies** for Strategies A through G were generated using `--dry-run` placeholder logic.
>
> Specifically:
> 1. All 113 prompts across Strategies A–G recorded a fixed synthetic execution latency of **`execution_latency_ms = 120.0 ms`**.
> 2. Model outputs were generated as synthetic placeholder strings (`"[DRY RUN SYNTHESIZED RESPONSE FOR ...]"`).
> 3. Quality scores were calculated by running the quality evaluator on these short dry-run placeholder strings.
> 4. As a direct consequence, the reported latency comparisons, quality comparisons, and $p = 0.0382$ statistical significance claims are **SYNTHETIC AND EMPIRICALLY INVALID**.
>
> **Safety Invariants Verified Intact**: `BaselineAdaptivePolicy` remains the sole production authority (`production_override = false`), `RLContextualBanditPolicy` remains SHADOW-ONLY, and `BAAI/bge-m3` (Action 7) remains masked.

---

## B. DATA PROVENANCE & SOURCE TRACING

### Latency Source Tracing
- **Code Origin**: `validation/runners/run_strategy_comparison.py` lines 95–101:
  ```python
  if dry_run:
      gen_text = f"[DRY RUN SYNTHESIZED RESPONSE FOR {p_id} using {model_id}]"
      lat_ms = 120.0
      success = True
      failover = False
  ```
- **Raw Log Verification**: Inspection of `validation/results/raw/raw_results_strategy_A.jsonl` through `raw_results_strategy_G.jsonl` confirms every entry contains `"execution_latency_ms": 120.0`.
- **Decision Latency Verification**: For Strategy G (`BaselineAdaptivePolicy`), decision latency was genuinely measured (e.g. `22,925.76 ms` initial embedding load, then `319.11 ms` to `602.87 ms` per decision). However, `total_latency_ms` added the synthetic `120.0 ms` execution placeholder.

---

## C. LIVE EXECUTION VERIFICATION

| Strategy ID | Strategy Name | Prompts | Actual Model Calls | Dry-Run Mocked Calls | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy A** | Fixed Gemma | 113 | 0 | 113 | `SYNTHETIC / DRY RUN` |
| **Strategy B** | Fixed Qwen | 113 | 0 | 113 | `SYNTHETIC / DRY RUN` |
| **Strategy C** | Fixed DeepSeek | 113 | 0 | 113 | `SYNTHETIC / DRY RUN` |
| **Strategy D** | Random Routing | 113 | 0 | 113 | `SYNTHETIC / DRY RUN` |
| **Strategy E** | Round-Robin | 113 | 0 | 113 | `SYNTHETIC / DRY RUN` |
| **Strategy F** | Capability Heuristic | 113 | 0 | 113 | `SYNTHETIC / DRY RUN` |
| **Strategy G** | BaselineAdaptivePolicy | 113 | 0 (Routing decision live) | 113 | `PARTIAL (Routing Live, Generation Synthetic)` |
| **Strategy H** | RL Shadow Evaluator | 790 (buffer) | 0 (Offline) | 0 | `VALID OFFLINE SHADOW EVALUATION` |

---

## D. LATENCY VALIDITY AUDIT

- **Classification**: **`INVALID`**
- **Reasoning**: Execution latency of 120.0ms is a hardcoded fallback value in `run_strategy_comparison.py`. Real end-to-end LLM text generation latencies (which range from 500ms to 40,000ms depending on local CPU vs cloud API) were not recorded.

---

## E. QUALITY SCORE VALIDITY AUDIT

- **Classification**: **`INVALID`**
- **Reasoning**: Quality scores (ranging from 3.16 to 3.17 / 5.0) were produced by passing dry-run placeholder text into `QualityMetricsEvaluator`. Because placeholder strings lack domain depth and length, the quality metrics do not reflect actual model output quality.
- **Evaluator Bias Check**: The evaluator itself (`QualityMetricsEvaluator`) is neutral and rule-based, but evaluated synthetic input.

---

## F. STATISTICAL VALIDITY AUDIT

- **Classification**: **`INVALID`**
- **Reasoning**: The reported $p = 0.0382$ and bootstrap confidence intervals were computed over synthetic quality scores and 120ms execution latencies. Statistical hypothesis testing over synthetic data yields invalid p-values.

---

## G. COST VALIDITY AUDIT

- **Classification**: **`VALID WITH LIMITATIONS`**
- **Reasoning**: Token pricing calculations ($/1k tokens for Gemini, Mistral, Groq, OpenRouter) are mathematically correct according to the configured pricing tables. However, input and output token counts were estimated from prompt string length and synthetic dry-run response length.

---

## H. RELIABILITY VALIDITY AUDIT

- **Classification**: **`INCONCLUSIVE`**
- **Reasoning**: Reported 100% success rate occurred because dry-run execution bypasses live HTTP/Ollama network calls and API rate limits. Real reliability and failover rates under 429 errors were not measured during this dry-run sweep.

---

## I. RESOURCE VALIDITY AUDIT

- **Classification**: **`VALID WITH LIMITATIONS`**
- **Reasoning**: Telemetry bounds (Ollama 1–3 local workers, cloud 4 workers) reflect actual system configuration settings in `ResourceAnalyzer` and `ParallelTaskScheduler`.

---

## J. RL IPS / SNIPS / ESS VALIDITY AUDIT

- **Classification**: **`VALID WITH LIMITATIONS (WEAK EVIDENCE)`**
- **Evidence Verification**:
  - **Propensity Records Evaluated**: Exactly **790 valid records** in `data/rl/experience_buffer.jsonl`.
  - **Policy Agreement Rate**: **59.93%**
  - **Estimated IPS Policy Value**: **1.1502**
  - **Estimated SNIPS Policy Value**: **0.9424**
  - **Effective Sample Size (ESS)**: **18.8** (Below required exit threshold of ESS $\ge 30$).
  - **Positivity Coverage**: **30.0%**
  - **Masking Check**: Action 7 (`BAAI/bge-m3`) was verified masked.
  - **Shadow Status**: Verified strictly shadow-only (`production_override = false`).
- **Conclusion**: The offline RL evaluation logic is mathematically sound and uses real propensity-logged buffer data. However, because $ESS = 18.8 < 30$, the empirical evidence is classified as **WEAK / INSUFFICIENT**.

---

## K. CLAIMS THAT MUST BE CORRECTED

The following claims in Reports 03–14 are overstated and MUST be explicitly flagged:
1. **"120ms Latency Across Strategies"** ➔ Correct to: *Synthetic dry-run placeholder value.*
2. **"Statistically Significant Quality Improvement ($p = 0.0382$)"** ➔ Correct to: *Invalid due to synthetic dry-run evaluation.*
3. **"100% Reliability & Success Rate"** ➔ Correct to: *Simulated dry-run outcome.*
4. **"RL Production Readiness"** ➔ Maintain classification: **`A. NOT READY — Insufficient ESS (18.8 < 30)`**.

---

## L. RESULTS THAT ARE VALID
1. **Dataset Integrity**: 113 prompts across 15 categories (v1.0.0).
2. **Routing Decision Latency**: `BaselineAdaptivePolicyAdapter` decision latency (~1.2ms to ~350ms).
3. **Safety Invariants**: `production_override = false`, `BaselineAdaptivePolicy` as authority, Action 7 masking.
4. **Offline RL IPS/SNIPS/ESS Math**: Calculated correctly from 790 real experience records ($ESS = 18.8$).

---

## M. RESULTS THAT ARE INVALID OR INCONCLUSIVE
1. **Model Generation Latencies**: Invalid (synthetic 120ms placeholder).
2. **Quality Scores & Rubric Ratings**: Invalid (computed on dry-run placeholder text).
3. **Statistical Significance ($p$-values)**: Invalid (computed on synthetic data).
4. **E2E Reliability Rates**: Inconclusive (bypassed network execution).

---

## N. REQUIRED RE-RUNS BEFORE ACADEMIC PUBLICATION / THESIS USE

To convert this validation into an empirically sound paper/thesis section, the following live runs must be executed:
1. Execute `python validation/runners/run_all_validations.py` **WITHOUT `--dry-run`** (or with active local/cloud LLM providers).
2. Record real end-to-end wall-clock generation latencies per model.
3. Record real LLM-generated output text for all 113 prompts.
4. Evaluate quality scores on real generated response text.
5. Re-compute bootstrap 95% CIs and paired t-tests over real quality and latency metrics.

---

## O. FINAL SCIENTIFIC INTEGRITY STATUS

| Component | Status |
| :--- | :--- |
| **Dataset Definition (113 Prompts, v1.0.0)** | **`VALID`** |
| **Safety Invariants (`production_override = false`)** | **`VALID`** |
| **Routing Decision Engine Logic** | **`VALID`** |
| **Model Text Generation Latency** | **`INVALID (SYNTHETIC DRY RUN)`** |
| **Model Output Quality Scores** | **`INVALID (SYNTHETIC DRY RUN)`** |
| **Statistical Significance ($p$-values)** | **`INVALID (SYNTHETIC DRY RUN)`** |
| **Offline RL Propensity & ESS Analysis** | **`VALID WITH LIMITATIONS (ESS = 18.8 < 30)`** |
| **Final RL Production Status** | **`NOT READY (SHADOW MODE ENFORCED)`** |
