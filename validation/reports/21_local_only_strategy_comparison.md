# 21. STRATEGY COMPARISON: LOCAL_ONLY_V1 (RUN 2)

**Experiment ID**: `RUN_2_LOCAL_ONLY_V1`  
**Focus**: Controlled Comparative Evaluation of Routing Strategies A–G in Local Execution Mode  

---

## 1. Comparative Performance Matrix

```
Quality (0-5)
  5.0 |                                                [C: 4.62]  [F: 4.57]
      |                                   [A: 4.56]
  4.5 |                     [D: 4.49]     [E: 4.43]    [G: 4.42]
      |      [B: 4.24]
  4.0 |____________________________________________________________________
      0s        10s           20s           30s          40s         70s
                                    Mean Latency (s)
```

| Strategy ID | Strategy Name | Mean Quality | Quality 95% CI | Mean Latency | Median Latency | P95 Latency | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy A** | Fixed Gemma (`gemma-3-4b`) | **4.56** | [4.50, 4.62] | 18.83s | 20.53s | 22.30s | **100%** |
| **Strategy B** | Fixed Qwen (`qwen-coder-3b`) | **4.24** | [4.14, 4.34] | **5.50s** | **2.96s** | **10.73s** | **100%** |
| **Strategy C** | Fixed DeepSeek (`deepseek-r1-7b`) | **4.62** | [4.55, 4.68] | 65.78s | 68.88s | 69.64s | **100%** |
| **Strategy D** | Local Random | **4.49** | [4.41, 4.57] | 36.00s | 26.36s | 80.54s | **100%** |
| **Strategy E** | Local Round-Robin | **4.43** | [4.34, 4.52] | 36.67s | 27.70s | 80.71s | **100%** |
| **Strategy F** | Local Capability Heuristic | **4.57** | [4.51, 4.63] | 24.21s | 20.51s | 71.33s | **100%** |
| **Strategy G** | BaselineAdaptivePolicy (Local) | **4.42** | [4.33, 4.50] | 32.46s | 19.46s | 81.04s | **100%** |

---

## 2. Key Strategy Takeaways

1. **Fixed Model Upper Bounds**:
   - **Fixed DeepSeek (C)** sets the quality ceiling at **4.62 / 5.0**, but at a steep latency penalty (65.78s).
   - **Fixed Qwen (B)** sets the latency floor at **5.50s**, providing rapid generation at acceptable quality (4.24).
   - **Fixed Gemma (A)** acts as a balanced default model (**4.56 / 5.0** quality at **18.83s** latency).

2. **Heuristic vs. Random Routing**:
   - **Strategy F (Capability Heuristic)** successfully routes coding/math tasks to Qwen (5.5s), complex reasoning to DeepSeek (65.8s), and general QA to Gemma (18.8s). This yields a high overall quality of **4.57 / 5.0** while keeping mean latency to **24.21s**.
   - **Strategy D (Random)** and **Strategy E (Round-Robin)** blindly pick models regardless of category, resulting in higher average latencies (~36.0s) without quality benefits.

3. **Strategy G (BaselineAdaptivePolicy in Local Mode)**:
   - Evaluates weighted feature scores over candidate models restricted strictly to local execution (`gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b`).
   - Achieves **4.42 / 5.0** quality at a median latency of **19.46s**.
