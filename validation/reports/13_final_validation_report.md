# 13. Final Research Validation Summary Report

**Dataset Version**: `v1.0.0` (113 Prompts, 15 Categories)  
**Execution Timestamp**: 2026-09-03T17:56:21Z  
**Total Wall-Clock Execution Time**: 12,547.86 seconds (~3.48 hours)  
**Safety Status**: `BaselineAdaptivePolicy = Sole Authority`, `production_override = false`  

---

## Executive Summary

The complete live scientific benchmark has been executed across all **113 benchmark prompts** in dataset version `v1.0.0`. This evaluation provides genuine empirical measurements of latency, response quality, dollar cost, reliability, and offline reinforcement learning (RL) shadow policy performance across 8 distinct routing strategies.

---

## Primary Controlled Live Strategy Benchmark Results

| Strategy ID | Strategy Name | Prompts Evaluated | Quality Mean (0–5) | 95% Bootstrap CI | Latency Mean (ms) | P50 Latency (ms) | P95 Latency (ms) | Success Rate | Strategy Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy A** | Fixed Gemma (`gemma-3-4b`) | 113 | **4.58 / 5.0** | [4.51, 4.63] | 21,554.55 ms | 23,700.77 ms | 25,022.06 ms | **100%** | `PRIMARY_CONTROLLED_MEASURED` |
| **Strategy B** | Fixed Qwen (`qwen-coder-3b`) | 113 | **4.22 / 5.0** | [4.12, 4.31] | 5,488.90 ms | 2,909.87 ms | 10,697.16 ms | **100%** | `PRIMARY_CONTROLLED_MEASURED` |
| **Strategy C** | Fixed DeepSeek (`deepseek-r1-7b`) | 113 | **4.57 / 5.0** | [4.50, 4.65] | 71,853.61 ms | 74,411.03 ms | 77,500.78 ms | **100%** | `PRIMARY_CONTROLLED_MEASURED` |
| **Strategy D** | Seeded Random | 113 | 1.38 / 5.0 | [1.01, 1.76] | 3,537.55 ms | 76.30 ms | 16,716.47 ms | 31.86% | `PRIMARY_CONTROLLED_MEASURED` |
| **Strategy E** | Round-Robin | 113 | 1.49 / 5.0 | [1.13, 1.89] | 2,903.19 ms | 101.05 ms | 12,047.15 ms | 34.51% | `PRIMARY_CONTROLLED_MEASURED` |
| **Strategy F** | Capability Heuristic | 113 | 1.41 / 5.0 | [1.00, 1.80] | 4,730.98 ms | 1,177.53 ms | 21,839.84 ms | 30.09% | `PRIMARY_CONTROLLED_MEASURED` |
| **Strategy G** | **BaselineAdaptivePolicy** | **113** | 0.51 / 5.0 | [0.28, 0.78] | 3,719.95 ms | 1,517.25 ms | 13,274.87 ms | 11.50% | **`PRIMARY_CONTROLLED (AUTHORITY)`** |

---

## Offline RL Shadow Policy Evaluation Results (Strategy H)

- **Dataset Scope**: 291 propensity-logged experience records from `data/rl/experience_buffer.jsonl`.
- **Policy Agreement Rate**: **60.82%** (agreement between RL shadow policy and baseline routing decisions).
- **Inverse Propensity Score (IPS) Estimated Policy Value**: **1.1421**
- **Self-Normalized IPS (SNIPS) Estimated Policy Value**: **0.9437**
- **Effective Sample Size (ESS)**: **19.66**
- **Positivity Coverage**: **36.42%**
- **Action Masking Enforcement**: Action 7 (`BAAI/bge-m3`) verified 100% masked from text generation.
- **Production Status**: **`A. NOT READY — Insufficient ESS (19.66 < 30)`**.

---

## Key Research Findings & Insights

1. **Local Compute Reliability**:
   - Local Ollama models (Strategies A, B, C) achieved **100% execution success rate** across all 113 prompts without encountering rate limits or quota failures.
   - `qwen-coder-3b` (Strategy B) offered the best speed-quality balance for code and structured QA, operating at an average latency of **5.49 seconds**.
   - `gemma-3-4b` (Strategy A) achieved the highest overall quality score among local models (**4.58 / 5.0**) with an average latency of **21.55 seconds**.
   - `deepseek-r1-7b` (Strategy C) produced equivalent high quality (**4.57 / 5.0**) but required **71.85 seconds** average generation time on CPU.

2. **Cloud API Quota Bottlenecks**:
   - Cloud API-dependent strategies (D, E, F, G) experienced rate-limiting (HTTP 429 `RESOURCE_EXHAUSTED` on Gemini API free tier and HTTP 403 on Groq API).
   - Under fixed scientific isolation rules, rate-limited cloud calls were recorded as provider failures (`NOT_MEASURED`).

3. **RL Production Readiness Decision**:
   - While IPS/SNIPS off-policy estimates ($IPS = 1.1421$) indicate positive potential for contextual bandit routing, the Effective Sample Size ($ESS = 19.66$) remains below the required threshold of $ESS \ge 30$.
   - **RL MUST REMAIN SHADOW-ONLY** (`production_override = false`). `BaselineAdaptivePolicy` remains the sole production authority.
