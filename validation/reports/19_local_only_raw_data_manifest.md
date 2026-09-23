# 19. RAW DATA MANIFEST: LOCAL_ONLY_V1 (RUN 2)

**Experiment ID**: `RUN_2_LOCAL_ONLY_V1`  
**Dataset Version**: `v1.0.0` (113 prompts across 15 categories)  
**Execution Environment**: Local Ollama Server (`http://localhost:11434`)  
**Hardware Platform**: CPU Execution (No CUDA GPU acceleration)  
**Output Location**: `validation/results/local_only/`  

---

## 1. Executive Summary

This manifest documents all 791 empirical telemetry records produced during the **`LOCAL_ONLY_V1` (Run 2)** benchmark. To eliminate cloud network latency, rate-limiting, and HTTP 429 quota exhaustion observed in Run 1, `LOCAL_ONLY_V1` restricted all text generation candidate models strictly to local Ollama models (`gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b`).

---

## 2. Dataset & Strategy Breakdown

| Strategy | Strategy Description | Target Model(s) | Dataset Prompts | Total Telemetry Records | Success Rate | Total Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy A** | Fixed Gemma | `gemma-3-4b` | 113 | 113 | **100%** (113/113) | **\$0.00** |
| **Strategy B** | Fixed Qwen | `qwen-coder-3b` | 113 | 113 | **100%** (113/113) | **\$0.00** |
| **Strategy C** | Fixed DeepSeek | `deepseek-r1-7b` | 113 | 113 | **100%** (113/113) | **\$0.00** |
| **Strategy D** | Local Random ($seed=42$) | `gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b` | 113 | 113 | **100%** (113/113) | **\$0.00** |
| **Strategy E** | Local Round-Robin | `gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b` | 113 | 113 | **100%** (113/113) | **\$0.00** |
| **Strategy F** | Local Capability Heuristic | `gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b` | 113 | 113 | **100%** (113/113) | **\$0.00** |
| **Strategy G** | BaselineAdaptivePolicy (Local) | `gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b` | 113 | 113 | **100%** (113/113) | **\$0.00** |
| **Total** | **Full Local Benchmark** | **3 Local Models** | **113** | **791** | **100% (791/791)** | **\$0.00** |

---

## 3. Raw Data Artifact File Map

All 791 individual raw telemetry JSON objects are stored chronologically in line-delimited JSONL format:

- `validation/results/local_only/raw/raw_results_strategy_A.jsonl` (113 records)
- `validation/results/local_only/raw/raw_results_strategy_B.jsonl` (113 records)
- `validation/results/local_only/raw/raw_results_strategy_C.jsonl` (113 records)
- `validation/results/local_only/raw/raw_results_strategy_D.jsonl` (113 records)
- `validation/results/local_only/raw/raw_results_strategy_E.jsonl` (113 records)
- `validation/results/local_only/raw/raw_results_strategy_F.jsonl` (113 records)
- `validation/results/local_only/raw/raw_results_strategy_G.jsonl` (113 records)

### Summary Manifest Files
- `validation/results/local_only/local_only_summary.json`
- `validation/results/local_only/local_only_manifest.json`

---

## 4. Telemetry Field Standard

Every JSON record in the raw dataset contains the following schema:
- `run_id`: `"RUN_2_LOCAL_ONLY_V1"`
- `experiment_condition`: `"LOCAL_ONLY_V1"`
- `strategy`: Strategy identifier (`"A"` through `"G"`)
- `prompt_id`: Unique benchmark prompt identifier (`"PROMPT-A01"` through `"PROMPT-O05"`)
- `original_prompt`: Text prompt string
- `task_category`: Category label (e.g., `Coding`, `Reasoning`, `Simple QA`, `Complex Multi-Objective`)
- `difficulty`: Prompt difficulty level (`easy`, `medium`, `hard`)
- `selected_model`: Exact local Ollama model ID used
- `provider`: `"Local Ollama"`
- `routing_decision_latency_ms`: Routing evaluation duration in ms
- `model_execution_latency_ms`: Model inference generation duration in ms
- `total_e2e_latency_ms`: Combined end-to-end latency in ms
- `success`: Boolean execution status (`true`)
- `failure_type`: `null`
- `failover_used`: `false`
- `input_tokens` / `output_tokens`: Measured token counts
- `total_cost_usd`: `0.0`
- `score_0_to_5`: Measured quality score on 0.0 – 5.0 scale
- `normalized_quality`: Measured quality score normalized on 0.0 – 1.0 scale
- `response_text_snippet`: First 200 characters of actual generated output text
