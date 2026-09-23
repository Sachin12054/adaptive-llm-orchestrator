# 22. TRADEOFF ANALYSIS: LOCAL_ONLY_V1 (RUN 2)

**Focus**: Multi-Objective Quality vs. Latency vs. Cost Trade-Off Analysis in Local Execution Mode  

---

## 1. Quality-Latency-Cost Trade-Off Analysis

In a 100% local execution environment, financial API cost is identically zero across all strategies ($Cost = \$0.00$). Therefore, the multi-objective optimization collapses to the fundamental trade-off between **Response Quality** and **Inference Latency**.

```
High Quality (4.62) |                        [C: DeepSeek R1]
                   |         [F: Heuristic]
                   | [A: Gemma 3]
                   | [D: Random]  [E: RoundRobin]  [G: Adaptive]
                   |
Low Quality (4.24)  | [B: Qwen Coder]
                   +--------------------------------------------
                     Fast (5.5s)     Medium (24.2s)   Slow (65.8s)
```

---

## 2. Pareto Efficiency Frontier

| Strategy ID | Strategy | Mean Quality (0-5) | Mean Latency (s) | Pareto Efficient? | Optimal Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B** | Fixed Qwen | **4.24** | **5.50s** | **YES (Latency Frontier)** | Ultra-low latency interactive code/QA |
| **A** | Fixed Gemma | **4.56** | **18.83s** | **YES (Balanced Frontier)** | General-purpose low-latency QA |
| **F** | Local Heuristic | **4.57** | **24.21s** | **YES (Quality/Speed Optimal)** | Category-aware task routing |
| **C** | Fixed DeepSeek | **4.62** | **65.78s** | **YES (Quality Frontier)** | Maximum quality offline reasoning |
| **D** | Local Random | 4.49 | 36.00s | **NO** | Dominated by A and F |
| **E** | Local RoundRobin | 4.43 | 36.67s | **NO** | Dominated by A and F |
| **G** | BaselineAdaptive | 4.42 | 32.46s | **NO** | Dominated by A and F |

---

## 3. Analytical Findings

1. **Latency Penalty of Reasoning**: DeepSeek R1 7B (Strategy C) provides a slight +0.06 quality gain over Strategy A (4.62 vs 4.56), but requires **3.49$\times$ higher latency** (65.78s vs 18.83s).
2. **Category Routing Optimization**: Strategy F (Capability Heuristic) captures **98.9% of DeepSeek's quality** (4.57 vs 4.62) while operating at **36.8% of its latency** (24.21s vs 65.78s).
3. **Speed Dominance of Small Models**: Qwen 2.5 Coder 3B (Strategy B) executes in **5.50 seconds** (11.9$\times$ faster than DeepSeek) while retaining acceptable quality (**4.24 / 5.0**).
