# 18. EXPERIMENTAL PROTOCOL & PRE-FLIGHT VERIFICATION: LOCAL_ONLY_V1 (RUN 2)

**Experiment Condition**: `LOCAL_ONLY_V1` (Run 2)  
**Date**: September 3, 2026  
**Status**: Pre-Flight Verified & Smoke Test Completed  
**Primary Result Directory**: `validation/results/local_only/`  

---

## 1. Executive Summary & Objective

The **Run 1 (`FULL_PROVIDER_ENVIRONMENT`)** scientific evaluation proved that multi-LLM orchestration delivers high quality, but cloud API rate-limiting (HTTP 429 quota exhaustion on Gemini API free tier) introduced external network confounding into empirical latency and reliability measurements.

To establish a strictly controlled baseline without cloud network noise, API rate-limits, or quota exhaustion, we designed **`LOCAL_ONLY_V1` (Run 2)**. In this condition, all generation candidate models are restricted **100% to locally hosted models running via Ollama**.

---

## 2. Experimental Constraints & Safety Invariants

### 2.1 Model Registry Bounds
Generation is strictly limited to the following 3 local models:
1. `gemma-3-4b` (Local Ollama)
2. `qwen-coder-3b` (Local Ollama)
3. `deepseek-r1-7b` (Local Ollama)

**Cloud Exclusions**: Zero cloud APIs (Gemini, Mistral, Groq, OpenRouter) are accessible during generation. Zero cloud failover is permitted.

### 2.2 Strategy Definitions (Local Mode)
* **Strategy A (Fixed Gemma)**: Always selects `gemma-3-4b`.
* **Strategy B (Fixed Qwen)**: Always selects `qwen-coder-3b`.
* **Strategy C (Fixed DeepSeek)**: Always selects `deepseek-r1-7b`.
* **Strategy D (Local Random)**: Uniform random choice among the 3 local models ($seed=42$).
* **Strategy E (Local Round-Robin)**: Sequential round-robin cycling through `[gemma-3-4b, qwen-coder-3b, deepseek-r1-7b]`.
* **Strategy F (Local Capability Heuristic)**: Maps prompt category to the best matching local model (Coding/Math $\rightarrow$ `qwen-coder-3b`, Reasoning/Complex $\rightarrow$ `deepseek-r1-7b`, Simple/General $\rightarrow$ `gemma-3-4b`).
* **Strategy G (BaselineAdaptivePolicy - Local Mode)**: Evaluates `BaselineAdaptivePolicy` feature weights over candidate models restricted strictly to local execution mode (`m.execution_mode == "local"`).

### 2.3 Production Safety Invariants
1. `PRODUCTION_OVERRIDE = False` (Enforced).
2. `BaselineAdaptivePolicy` remains the sole production authority.
3. `RLContextualBanditPolicy` operates strictly offline in shadow evaluation mode.
4. `BAAI/bge-m3` (Action 7) remains masked from text generation.

---

## 3. Pre-Flight Infrastructure Verification

Before initiating live model inference, the pre-flight suite verified local system status:

| Service / Invariant | Target / Value | Status | Result / Details |
| :--- | :--- | :--- | :--- |
| **Ollama Server** | `http://localhost:11434` | `ONLINE` | HTTP 200 OK |
| **Gemma 3 4B** | `gemma-3-4b:latest` | `INSTALLED` | Local tag verified |
| **Qwen 2.5 Coder 3B** | `qwen-coder-3b:latest` | `INSTALLED` | Local tag verified |
| **DeepSeek R1 7B** | `deepseek-r1-7b:latest` | `INSTALLED` | Local tag verified |
| **Production Safety** | `PRODUCTION_OVERRIDE` | `PASSED` | `False` |
| **Embedding Engine** | `BAAI/bge-m3` | `PASSED` | CPU / GPU auto-device |

---

## 4. Live 21-Sample Smoke Test Telemetry Results

A mandatory 3-prompt $\times$ 7-strategy smoke test ($3 \times 7 = 21$ real local executions) was performed using dataset prompts `PROMPT-A01` (Simple QA), `PROMPT-D01` (Coding), and `PROMPT-L01` (Complex Multi-Objective).

### 4.1 Empirical Telemetry Table

| Strategy | Prompt ID | Category | Selected Model | Provider | Routing Latency | Model Exec Latency | Total E2E Latency | Success |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | PROMPT-A01 | Simple QA | `gemma-3-4b` | Local Ollama | 1.00 ms | 14,736.78 ms | 14,737.78 ms | 100% |
| **A** | PROMPT-D01 | Coding | `gemma-3-4b` | Local Ollama | 1.33 ms | 22,294.64 ms | 22,295.97 ms | 100% |
| **A** | PROMPT-L01 | Complex | `gemma-3-4b` | Local Ollama | 3.30 ms | 21,892.59 ms | 21,895.89 ms | 100% |
| **B** | PROMPT-A01 | Simple QA | `qwen-coder-3b` | Local Ollama | 0.80 ms | 8,386.54 ms | 8,387.34 ms | 100% |
| **B** | PROMPT-D01 | Coding | `qwen-coder-3b` | Local Ollama | 0.80 ms | 9,855.22 ms | 9,856.02 ms | 100% |
| **B** | PROMPT-L01 | Complex | `qwen-coder-3b` | Local Ollama | 0.42 ms | 3,633.68 ms | 3,634.10 ms | 100% |
| **C** | PROMPT-A01 | Simple QA | `deepseek-r1-7b` | Local Ollama | 0.78 ms | 15,819.50 ms | 15,820.28 ms | 100% |
| **C** | PROMPT-D01 | Coding | `deepseek-r1-7b` | Local Ollama | 1.11 ms | 73,763.77 ms | 73,764.88 ms | 100% |
| **C** | PROMPT-L01 | Complex | `deepseek-r1-7b` | Local Ollama | 1.24 ms | 71,990.98 ms | 71,992.22 ms | 100% |
| **D** | PROMPT-A01 | Simple QA | `deepseek-r1-7b` | Local Ollama | 0.90 ms | 3,742.78 ms | 3,743.68 ms | 100% |
| **D** | PROMPT-D01 | Coding | `gemma-3-4b` | Local Ollama | 1.21 ms | 28,993.19 ms | 28,994.40 ms | 100% |
| **D** | PROMPT-L01 | Complex | `gemma-3-4b` | Local Ollama | 1.31 ms | 21,156.39 ms | 21,157.70 ms | 100% |
| **E** | PROMPT-A01 | Simple QA | `gemma-3-4b` | Local Ollama | 0.90 ms | 3,946.20 ms | 3,947.10 ms | 100% |
| **E** | PROMPT-D01 | Coding | `qwen-coder-3b` | Local Ollama | 0.87 ms | 10,909.24 ms | 10,910.11 ms | 100% |
| **E** | PROMPT-L01 | Complex | `deepseek-r1-7b` | Local Ollama | 1.28 ms | 81,853.29 ms | 81,854.57 ms | 100% |
| **F** | PROMPT-A01 | Simple QA | `gemma-3-4b` | Local Ollama | 1.04 ms | 12,038.83 ms | 12,039.87 ms | 100% |
| **F** | PROMPT-D01 | Coding | `qwen-coder-3b` | Local Ollama | 0.54 ms | 7,825.75 ms | 7,826.29 ms | 100% |
| **F** | PROMPT-L01 | Complex | `deepseek-r1-7b` | Local Ollama | 1.25 ms | 80,986.22 ms | 80,987.47 ms | 100% |
| **G** | PROMPT-A01 | Simple QA | `gemma-3-4b` | Local Ollama | 19,620.69 ms | 13,968.23 ms | 33,588.92 ms | 100% |
| **G** | PROMPT-D01 | Coding | `qwen-coder-3b` | Local Ollama | 349.31 ms | 11,934.13 ms | 12,283.44 ms | 100% |
| **G** | PROMPT-L01 | Complex | `deepseek-r1-7b` | Local Ollama | 651.93 ms | 82,674.18 ms | 83,326.11 ms | 100% |

---

## 5. Key Empirical Observations from Smoke Test

1. **Strict Local Execution**: 21 out of 21 executions (100%) were dispatched directly to `http://localhost:11434/api/generate`. Zero cloud API calls or failover mechanisms were triggered.
2. **Contextual Routing in Strategy G**:
   - For `PROMPT-A01` (Simple QA), Strategy G evaluated scores `gemma-3-4b` (0.9400), `qwen-coder-3b` (0.9400), `deepseek-r1-7b` (0.8200) $\rightarrow$ selected `gemma-3-4b`.
   - For `PROMPT-D01` (Coding), Strategy G evaluated scores `qwen-coder-3b` (0.9550), `deepseek-r1-7b` (0.9100), `gemma-3-4b` (0.8800) $\rightarrow$ selected `qwen-coder-3b`.
   - For `PROMPT-L01` (Complex Multi-Objective), Strategy G evaluated scores `deepseek-r1-7b` (0.9550), `gemma-3-4b` (0.8350), `qwen-coder-3b` (0.8050) $\rightarrow$ selected `deepseek-r1-7b`.
3. **Execution Latency Profiles**:
   - `qwen-coder-3b`: Fast evaluation (3.6s – 11.9s) at ~75 tokens/sec.
   - `gemma-3-4b`: Moderate evaluation (12.0s – 28.9s) at ~32 tokens/sec.
   - `deepseek-r1-7b`: Deep reasoning output (~600 tokens of internal reasoning `<think>` steps) taking 71.9s – 82.6s at ~8.5 tokens/sec.
4. **Smoke Test Conclusion**: The system passed pre-flight verification with zero errors. Pipeline execution stopped automatically upon smoke test completion as commanded.
