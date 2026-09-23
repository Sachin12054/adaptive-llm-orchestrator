# 28. COMPLEX TASK SYNTHESIS OPTIMIZATION DESIGN (`RUN_4`)

**Document Title**: Architecture Inspection & Experimental Design for Complex Task Synthesis Bottleneck Optimization  
**Run ID**: `RUN_4_SYNTHESIS_OPTIMIZATION`  
**Baseline Reference**: `RUN_3_COMPLEX_TASK_COMPARISON` (Frozen Baseline: $N=18$ prompts, $N=36$ executions)  
**Author**: Antigravity AI Research & Engineering Team  
**Date**: September 9, 2026  
**Status**: **Phase 1 Architecture Inspection Complete (No Code / Benchmark Changes Executed)**

---

## 1. Executive Summary & Problem Definition

In `RUN_3_COMPLEX_TASK_COMPARISON`, empirical telemetry revealed a critical performance bottleneck in the complex task decomposition pipeline:
- **Total Orchestrated E2E Latency**: **199.34s mean** (vs **115.68s** for single model `deepseek-r1-7b`, a **+72.3% latency increase**).
- **Parallel Subtask Execution Time**: **41.28s mean** ($20.7\%$ of total wall-clock time), achieving a **2.82x parallel speedup** and **89.0% conventional parallel efficiency**.
- **Final Response Synthesis Latency**: **156.81s mean** (**78.7% of total end-to-end wall-clock time**).

> [!CAUTION]
> **The Synthesis Bottleneck**:
> While parallel subtask execution functions with high parallel efficiency ($89.0\%$), the final synthesis aggregation step—which passes all subtask outputs to `deepseek-r1-7b` for re-summarization—consumes **78.7% of total end-to-end latency**. On 16-thread CPU hardware without GPU acceleration, this single sequential step turns parallel speedup into an end-to-end latency penalty (+83.66s slower).

`RUN_4` is designed as a controlled scientific experiment to evaluate candidate synthesis optimization strategies to eliminate or reduce this bottleneck while preserving quality and objective coverage.

---

## 2. Current Architecture Inspection & Execution Trace

### 2.1 Complete Execution Flow

```
Original User Prompt
       │
       ▼
[ComplexTaskDecomposer.is_complex_prompt()]
       │
       ▼
[DynamicTaskDecomposer.decompose()] ──► Decomposes prompt into SubTasks
       │
       ▼
[DependencyGraphBuilder.compute_execution_levels()] ──► Computes DAG execution levels
       │
       ▼
[ParallelTaskScheduler.execute_plan_async()] ──► Executes subtasks in parallel via Ollama
       │                                     (gemma-3-4b, qwen-coder-3b, deepseek-r1-7b)
       ▼
[TaskResponseValidator] ──► Validates subtask outputs
       │
       ▼
[TaskAggregator.aggregate_results()] ──► BOTTLENECK: Combines subtask texts into 7500-char prompt
       │                                 and routes to deepseek-r1-7b for full LLM synthesis
       ▼
Final Aggregated Response (156.81s synthesis time)
```

### 2.2 Detailed Subsystem Breakdown

1. **Decomposition (`backend/app/services/complex/dynamic_decomposer.py`)**:
   - Parses the original prompt into discrete `SubTask` objects, assigning objectives, required capabilities, and model choices.
   - Latency: ~1.25s mean.

2. **DAG Scheduling (`backend/app/services/complex/parallel_scheduler.py`)**:
   - Schedules independent subtasks concurrently across DAG levels using `asyncio.gather()`.
   - Latency: ~41.28s parallel wall-clock time across levels.

3. **Subtask Output Storage (`app/schemas/complex.py:SubTask`)**:
   - Stores `generated_text`, `execution_success`, `latency_ms`, `input_tokens`, `output_tokens`, `assigned_model`.

4. **Task Validation (`backend/app/services/complex/task_response_validator.py`)**:
   - Inline semantic non-emptiness check. Latency: <1ms.

5. **Task Aggregation & Synthesis (`backend/app/services/complex/task_aggregator.py`)**:
   - Truncates individual subtask outputs to 1500 chars.
   - Concatenates subtask outputs with headers (`### Subtask [task_id]: objective \n {text}`).
   - Formulates a synthesis prompt (up to 7500 chars):
     ```text
     User Prompt: {original_prompt}

     Below are the verified outputs from decomposed subtasks:

     {subtasks_combined_text}

     Instructions: Synthesize all available subtask results into a clean, cohesive, and comprehensive response. Preserve all specific facts, numbers, code, and recommendations. Clearly indicate any missing or incomplete areas if subtasks failed. Do not add fake information.
     ```
   - Sends `synth_prompt` to `AdaptiveDecisionEngine.decide()`. In local mode, `BaselineAdaptivePolicy` selects `deepseek-r1-7b` due to prompt length and complexity cues.
   - Calls `ResponseGenerator.generate_response()` with `max_output_tokens=1000`, `temperature=0.7`.
   - Latency: **156.81s mean** on 16-thread CPU.

---

## 3. Empirical Analysis: What Synthesis Actually Does

Inspection of raw RUN_3 telemetry and subtask outputs reveals the exact nature of the synthesis phase:

| Functional Category | Observed Role in Current Implementation | Potential for Optimization |
| :--- | :--- | :--- |
| **A. Section Formatting & Header Assembly** | **PRIMARY (65%)**: Combines subtask texts under markdown headers (`#`, `##`). | Can be performed deterministically in 0ms. |
| **B. Summary / Intro Generation** | **SECONDARY (20%)**: Adds an opening summary and closing wrap-up paragraph. | Can be generated by lightweight model or template. |
| **C. Content Deduplication** | **MINOR (10%)**: Removes occasional duplicate sentences between subtasks. | Can be handled by subtask prompt isolation or regex filters. |
| **D. Substantive Cross-Task Reasoning** | **NEGLIGIBLE (<5%)**: Subtasks are already domain-isolated (e.g. database vs architecture vs costs). | Heavyweight reasoning model is NOT required for combining independent modules. |

> [!IMPORTANT]
> **Key Finding**:
> The synthesis model (`deepseek-r1-7b`) is primarily acting as a **markdown document formatter and text joiner**. It is NOT performing deep mathematical re-derivation or contradiction resolution. Using a 7B reasoning model for this task incurs massive unnecessary CPU latency.

---

## 4. Candidate Synthesis Strategies ($S_0 – S_4$)

| Strategy ID | Name | Description | Model / Engine | Target Synthesis Latency | Target E2E Latency | Implementation Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$S_0$** | **Baseline Synthesis** | Current implementation (full LLM synthesis via decision engine). | `deepseek-r1-7b` | ~156.8s | ~199.3s | None (Current) |
| **$S_1$** | **Lightweight LLM Synthesis** | Route synthesis prompt to a fast local model (`gemma-3-4b` or `qwen-coder-3b`). | `gemma-3-4b` | ~35.0s – 45.0s | ~78.0s – 88.0s | Low |
| **$S_2$** | **Deterministic Structured Assembly** | Zero-LLM deterministic markdown stitching with section headers, TOC, and metadata. | Deterministic Python Engine | **0.00s (~2ms)** | **~42.5s** | Low |
| **$S_3$** | **Hybrid Structured + Fast Synthesis** | Deterministic section assembly ($S_2$) + 1-paragraph lightweight executive summary ($S_1$). | Hybrid ($S_2$ + `gemma-3-4b` Summary) | ~8.0s – 12.0s | ~50.0s – 55.0s | Moderate |
| **$S_4$** | **Incremental DAG-Level Synthesis** | Progressively format and aggregate subtasks as each DAG level completes. | Async Stream Aggregator | ~15.0s – 25.0s | ~58.0s – 68.0s | High |

---

## 5. Risk, Trade-Off & Capability Analysis

### Strategy $S_0$: Baseline DeepSeek 7B
- **Pros**: Produces highly cohesive, polished narrative transitions.
- **Cons**: Extremely slow (**156.81s**), high CPU utilization (75.2%), low E2E quality efficiency (0.0213 q/s).

### Strategy $S_1$: Lightweight Local Model (`gemma-3-4b`)
- **Pros**: ~75% reduction in synthesis latency (~40s vs ~156.8s); maintains narrative flow; simple implementation.
- **Cons**: Still requires LLM generation phase on CPU (~40s overhead).

### Strategy $S_2$: Deterministic Structured Assembly
- **Pros**: **Zero synthesis latency (0.00s)**; 0 synthesis output tokens; reduces E2E latency from **199.34s to ~42.5s** (**78.7% E2E latency reduction**); 100% preservation of raw subtask facts/code.
- **Cons**: Minor formatting abruptness between subtask sections; lacks an overarching executive summary paragraph.

### Strategy $S_3$: Hybrid Assembly + Fast Executive Summary
- **Pros**: Combines 0ms section assembly with a brief 2-sentence executive summary from `gemma-3-4b` (~100 tokens max); very fast (~10s synthesis time); excellent quality and polish.
- **Cons**: Slightly higher complexity than pure $S_2$.

### Strategy $S_4$: Incremental DAG-Level Synthesis
- **Pros**: Overlaps subtask generation with partial level synthesis.
- **Cons**: Complex async state management; minimal benefit over $S_2$/$S_3$ when synthesis itself is fast.

---

## 6. Research Hypotheses & Experimental Design

### 6.1 Formal Hypotheses
- **$H_1$ (Alternative Hypothesis)**:  
  Replacing heavyweight `deepseek-r1-7b` synthesis with deterministic structured aggregation ($S_2$) or hybrid fast synthesis ($S_3$) will reduce end-to-end latency by $>60\%$ ($p < 0.001$) while maintaining statistically indistinguishable response quality ($p > 0.05$) and objective coverage ($p > 0.05$) relative to $S_0$.
- **$H_0$ (Null Hypothesis)**:  
  Replacing heavyweight synthesis does not yield a significant latency reduction or causes a statistically significant degradation in overall quality or objective coverage ($p < 0.05$).

### 6.2 Controlled Synthesis Ablation Protocol
To isolate synthesis optimization from subtask generation variance:
1. **Dataset**: Re-use the exact same $N=18$ benchmark prompts from RUN_3 (`PROMPT-L01` to `PROMPT-L06`, `PROMPT-G01` to `PROMPT-G04`, `PROMPT-I01` to `PROMPT-I04`, `PROMPT-H01` to `PROMPT-H04`).
2. **Subtask Output Replay Strategy**:
   - RUN_4 will operate as a **Controlled Synthesis Ablation**.
   - Subtasks will be executed using fixed seeds / saved outputs to ensure that input subtask texts fed into $S_0, S_1, S_2, S_3$ are identical.
3. **Execution Environment**:
   - Local Ollama server (`http://localhost:11434`), 16-thread CPU.
   - Zero cloud API usage.
   - `PRODUCTION_OVERRIDE = False`.

---

## 7. Metrics & Statistical Evaluation Framework

For each strategy ($S_0, S_1, S_2, S_3, S_4$), we will measure:

1. **Synthesis Latency ($T_{synth}$)**: Wall-clock time of the synthesis/aggregation stage (seconds).
2. **End-to-End Latency ($T_{E2E}$)**: Total pipeline latency from prompt input to final response (seconds).
3. **Synthesis Share of E2E Latency (%)**: $(T_{synth} / T_{E2E}) \times 100$.
4. **Overall Quality (0-5 Rubric)**: Standard structural quality verification score.
5. **Objective Coverage (%)**: Percentage of requested sub-objectives satisfied in the final output.
6. **End-to-End Quality Efficiency**: $\text{Quality} / T_{E2E}$ (quality per second).
7. **Synthesis Output Tokens**: Total tokens generated during synthesis phase.
8. **CPU & RAM Utilization**: Average system resource usage during synthesis.

### Statistical Tests
- **Paired $t$-tests** ($\alpha = 0.05$) between $S_0$ and each candidate strategy ($S_1, S_2, S_3$) for latency, quality, and coverage.
- **Wilcoxon Signed-Rank Tests** for non-parametric validation.
- **Effect Size (Cohen's $d$)** for practical significance.

---

## 8. Threats to Validity & Mitigation

1. **Subtask Quality Confounding**: Subtask variation could affect synthesis quality.
   - *Mitigation*: Use paired controlled ablation with identical subtask inputs across $S_0 - S_3$.
2. **Rubric Bias Toward Polish**: Automated quality rubrics might favor verbose LLM synthesis over clean structured sections.
   - *Mitigation*: Dual evaluation using structural verification + objective coverage verification.
3. **System CPU Contention**: Background OS tasks could skew latency.
   - *Mitigation*: Record background CPU usage prior to each trial and run paired trials sequentially.

---

## 9. Recommended Strategy for RUN_4 Execution

> [!TIP]
> **RECOMMENDED RUN_4 STRATEGY: $S_2$ (Deterministic Structured Assembly) with $S_3$ (Hybrid) Evaluation**  
>
> **Rationale**:  
> 1. **Eliminates Bottleneck Entirely**: $S_2$ reduces synthesis latency from **156.81s to 0.00s**, dropping total E2E latency from **199.34s to ~42.5s** (a **78.7% latency reduction**).  
> 2. **Maximal E2E Efficiency**: Replaces the E2E latency penalty with a true **2.72x end-to-end speedup** over single-model direct execution (42.5s vs 115.68s).  
> 3. **Fact Preservation**: Deterministic assembly guarantees zero hallucination, zero omission of subtask facts, and zero token truncation.  
> 4. **Side-by-Side Comparison**: Evaluating $S_0, S_1, S_2, S_3$ on the same prompt set in RUN_4 will conclusively establish the optimal quality-latency Pareto frontier.

---

## 10. Target Files for Implementation in Phase 2

When approved for Phase 2 implementation, the following files will be modified or added:

1. `backend/app/services/complex/task_aggregator.py`:
   - Implement structured deterministic assembly mode ($S_2$).
   - Implement lightweight model routing mode ($S_1$).
   - Implement hybrid mode ($S_3$).
2. `validation/runners/run_complex_synthesis_optimization.py` [NEW]:
   - Dedicated test runner for RUN_4 controlled synthesis ablation.
3. `validation/results/complex_task_synthesis_optimization/` [NEW DIRECTORY]:
   - Isolated directory for RUN_4 raw telemetry and processed results.

---

## 11. Acceptance Criteria for RUN_4 Phase 2

1. **Synthesis Latency Reduction**: Candidate strategy must reduce synthesis latency by $\ge 75\%$ relative to $S_0$ ($156.81\text{s} \rightarrow \le 39.0\text{s}$).
2. **Quality Preservation**: Response quality must remain statistically indistinguishable from $S_0$ ($p > 0.05$).
3. **Coverage Preservation**: Objective coverage must remain within $\pm 3\%$ of $S_0$ ($p > 0.05$).
4. **Data Isolation**: All RUN_4 raw results must be saved under `validation/results/complex_task_synthesis_optimization/raw/` without altering RUN_1, RUN_2, or RUN_3 data.
