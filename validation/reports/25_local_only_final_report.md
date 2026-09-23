# 25. FINAL SCIENTIFIC REPORT: LOCAL_ONLY_V1 (RUN 2)

**Experiment Title**: Adaptive Multi-LLM Orchestration Under Local Model Execution (`LOCAL_ONLY_V1`)  
**Run ID**: `RUN_2_LOCAL_ONLY_V1`  
**Dataset Version**: `v1.0.0` (113 benchmark prompts, 15 task categories)  
**Total Telemetry Sample Size**: $N = 791$ real executions ($113 \text{ prompts} \times 7 \text{ strategies}$)  
**Total Benchmark Duration**: 24,174.96 seconds (~6.71 hours)  
**Overall System Reliability**: **100% Success Rate (791 / 791)**  
**Safety Invariant Compliance**: **100% Verified** (`PRODUCTION_OVERRIDE = False`)  

---

## 1. Executive Summary

This report completes the scientific benchmarking phase for **`LOCAL_ONLY_V1` (Run 2)**. To eliminate cloud API quota exhaustion, rate limiting, and network variability observed during Run 1 (`FULL_PROVIDER_ENVIRONMENT`), `LOCAL_ONLY_V1` evaluated Strategies A–G under strictly local Ollama model execution (`gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b`).

Over 6.71 hours of continuous live inference, the pipeline executed all 791 runs with **100% success rate**, zero failover, and zero financial cost (\$0.00 USD).

---

## 2. Final Empirical Master Benchmark Table

| Strategy ID | Strategy Name | Models Selected | Sample Count | Success Rate | Mean Quality (0-5) | 95% Quality CI | Mean Latency | Median Latency | P95 Latency | Financial Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy A** | Fixed Gemma | `gemma-3-4b` | 113 | **100%** | **4.56** | [4.50, 4.62] | 18.83s | 20.53s | 22.30s | **\$0.00** |
| **Strategy B** | Fixed Qwen | `qwen-coder-3b` | 113 | **100%** | **4.24** | [4.14, 4.34] | **5.50s** | **2.96s** | **10.73s** | **\$0.00** |
| **Strategy C** | Fixed DeepSeek | `deepseek-r1-7b` | 113 | **100%** | **4.62** | [4.55, 4.68] | 65.78s | 68.88s | 69.64s | **\$0.00** |
| **Strategy D** | Local Random | All 3 Local Models | 113 | **100%** | **4.49** | [4.41, 4.57] | 36.00s | 26.36s | 80.54s | **\$0.00** |
| **Strategy E** | Local Round-Robin | All 3 Local Models | 113 | **100%** | **4.43** | [4.34, 4.52] | 36.67s | 27.70s | 80.71s | **\$0.00** |
| **Strategy F** | Local Capability Heuristic | All 3 Local Models | 113 | **100%** | **4.57** | [4.51, 4.63] | 24.21s | 20.51s | 71.33s | **\$0.00** |
| **Strategy G** | BaselineAdaptivePolicy | All 3 Local Models | 113 | **100%** | **4.42** | [4.33, 4.50] | 32.46s | 19.46s | 81.04s | **\$0.00** |

---

## 3. Key Scientific Discoveries

### 3.1 Quality-Speed Pareto Efficiency
- **Fixed Qwen (Strategy B)** is the fastest local model (**5.50s mean latency**, ~75 tokens/sec), delivering respectable quality (**4.24 / 5.0**).
- **Fixed DeepSeek (Strategy C)** achieves the highest quality ceiling (**4.62 / 5.0**), but requires ~65.78s due to internal reasoning chains (`<think>` blocks).
- **Strategy F (Capability Heuristic)** achieves **98.9% of DeepSeek's quality** (**4.57 / 5.0**) at **36.8% of its latency** (**24.21s**).

### 3.2 Offline RL Shadow Evaluation (Strategy H)
- Evaluated over 291 propensity records: $IPS = 1.1421$, $SNIPS = 0.9437$, $ESS = 19.66$.
- Because $ESS = 19.66 < 30.0$, safety policy invariants strictly mandated keeping `RLContextualBanditPolicy` in offline shadow mode (`PRODUCTION_OVERRIDE = False`).

---

## 4. Final Conclusion

The **`LOCAL_ONLY_V1` benchmark (Run 2)** establishes an empirical baseline for multi-LLM orchestration without external API rate limits or network noise. All raw logs, summary statistics, figures, and research reports 18–25 are archived under `validation/results/local_only/` and `validation/reports/`.
