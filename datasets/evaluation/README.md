# Benchmark 100 Dataset (`benchmark_100.json`)

## 1. Dataset Purpose
The `benchmark_100.json` dataset provides a standardized, reproducible benchmark suite of 100 prompts for evaluating the performance, cost-efficiency, hardware sensitivity, intent accuracy, complexity classification accuracy, and model routing decisions of the **Adaptive Multi-LLM Orchestrator**.

---

## 2. Complexity Categories
The dataset contains exactly 100 prompts balanced equally across 4 complexity tiers (25 prompts per tier):

1. **`low` (25 Prompts)**: Factual questions, simple translations, direct syntax queries, short summaries, and basic single-operation tasks (Targeting low-cost/fast local models like `gemma-3-4b` or `gemini-3.5-flash`).
2. **`medium` (25 Prompts)**: Algorithmic explanations, code snippet debugging, multi-step math problems, trade-off evaluations, and structured translations (Targeting medium-capacity models like `qwen-coder-3b` or `mistral-small-latest`).
3. **`high` (25 Prompts)**: Data structure derivations, advanced linear algebra/calculus, deep architectural trade-off analyses, concurrency & deadlock investigations, and complex code refactoring (Targeting high-capacity reasoning models like `deepseek-r1-7b` or `llama-3.3-70b-versatile`).
4. **`very_high` (25 Prompts)**: Distributed system architecture blueprints, CAP theorem & Paxos derivations, lock-free C++ data structures, compiler JIT optimizations, and full-stack enterprise orchestration designs (Targeting frontier 70B parameter models like `meta-llama/llama-3.3-70b-instruct`).

---

## 3. Intent Categories
The dataset covers the 9 intent categories supported by the orchestrator:
1. `factual` (Factual QA)
2. `coding` (Coding & Software Engineering)
3. `mathematics` (Mathematics & Calculus)
4. `reasoning` (Logic, Architecture & System Reasoning)
5. `explanation` (Technical Concepts & Explanations)
6. `summarization` (Text Summarization)
7. `translation` (Multi-Lingual Translation)
8. `creative` (Creative Writing & Screenplays)
9. `conversational` (Multi-Turn & Dialogue Greetings)

---

## 4. Generation Methodology
* **Conceptual Difficulty Variance**: Prompts represent genuine variations in underlying problem difficulty and reasoning depth rather than artificial text length inflation.
* **Format & Machine-Readable Schema**:
  ```json
  {
    "id": "bench-001",
    "prompt": "What is the capital of France?",
    "intent": "factual",
    "expected_complexity": "low"
  }
  ```
* **No Manual Model Assignments**: Model selection decisions are strictly computed dynamically by the `BaselineAdaptivePolicy` routing engine based on prompt complexity, candidate capability, context window limits, token pricing, and live host hardware telemetry.

---

## 5. How to Reproduce the Benchmark
To execute the benchmark suite through the active orchestrator pipeline and record transition logs:

```bash
# Run the evaluation script
python evaluation/run_benchmark_100.py
```

Results will be stored in JSON Lines format at:
`data/evaluation/benchmark_100_results.jsonl`
