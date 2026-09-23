# 20. STATISTICAL ANALYSIS: LOCAL_ONLY_V1 (RUN 2)

**Experiment Condition**: `LOCAL_ONLY_V1` (Run 2)  
**Sample Size**: $N = 791$ total executions ($113 \text{ prompts} \times 7 \text{ strategies}$)  
**Evaluation Scope**: Latency, Quality, Cost, and Reliability across Local Ollama Models  

---

## 1. Executive Summary Table

| Strategy ID | Strategy Description | Model(s) Selected | Success Rate | Mean Quality (0-5) | 95% Quality CI | Mean Latency (s) | P50 Latency (s) | P95 Latency (s) | Avg Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy A** | Fixed Gemma | `gemma-3-4b` | **100%** | **4.56** | [4.50, 4.62] | 18.83s | 20.53s | 22.30s | **\$0.00** |
| **Strategy B** | Fixed Qwen | `qwen-coder-3b` | **100%** | **4.24** | [4.14, 4.34] | **5.50s** | **2.96s** | **10.73s** | **\$0.00** |
| **Strategy C** | Fixed DeepSeek | `deepseek-r1-7b` | **100%** | **4.62** | [4.55, 4.68] | 65.78s | 68.88s | 69.64s | **\$0.00** |
| **Strategy D** | Local Random | All 3 Local Models | **100%** | **4.49** | [4.41, 4.57] | 36.00s | 26.36s | 80.54s | **\$0.00** |
| **Strategy E** | Local Round-Robin | All 3 Local Models | **100%** | **4.43** | [4.34, 4.52] | 36.67s | 27.70s | 80.71s | **\$0.00** |
| **Strategy F** | Local Capability Heuristic | All 3 Local Models | **100%** | **4.57** | [4.51, 4.63] | 24.21s | 20.51s | 71.33s | **\$0.00** |
| **Strategy G** | BaselineAdaptivePolicy | All 3 Local Models | **100%** | **4.42** | [4.33, 4.50] | 32.46s | 19.46s | 81.04s | **\$0.00** |

---

## 2. Statistical Analysis & Confidence Intervals

### 2.1 Non-Parametric Bootstrap Quality Analysis (1,000 Resamples)
- **Strategy C (`deepseek-r1-7b`)** achieved the highest mean quality score of **4.62 / 5.0** (95% CI: [4.55, 4.68]), driven by explicit chain-of-thought reasoning (`<think>` blocks).
- **Strategy F (Local Capability Heuristic)** achieved **4.57 / 5.0** (95% CI: [4.51, 4.63]), statstically indistinguishable from Fixed DeepSeek ($p > 0.05$), while reducing mean latency by **63.2%** (24.21s vs 65.78s).
- **Strategy B (`qwen-coder-3b`)** traded slight quality for maximum speed (**4.24 / 5.0**, 95% CI: [4.14, 4.34]), executing in just **5.50 seconds** average wall-clock time.

### 2.2 Latency Distribution Analysis
- **Fixed Qwen (Strategy B)**: Extremely fast CPU inference (**5.50s mean, 2.96s median**), delivering ~75 tokens/second.
- **Fixed Gemma (Strategy A)**: Predictable, moderate latency (**18.83s mean, 20.53s median**), delivering ~32 tokens/second.
- **Fixed DeepSeek (Strategy C)**: High-latency reasoning (**65.78s mean, 68.88s median**), delivering ~8.5 tokens/second due to 600-token internal reasoning chains.
