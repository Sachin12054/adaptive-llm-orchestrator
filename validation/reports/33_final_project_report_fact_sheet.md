# 33. AUTHORITATIVE PROJECT REPORT FACT SHEET
## Adaptive Multi-LLM Orchestration Using Contextual Bandits

**Document Status**: Immutable Authoritative Fact Sheet  
**Target Output**: IEEE Overleaf LaTeX Report Generation  
**Project Repository**: `adaptive-llm-orchestrator`  
**Date of Audit**: September 2026  
**Data Integrity**: SHA-256 Verified across `RUN_1` to `RUN_4` Raw Telemetry  

---

## 1. Project Identity

- **Exact Project Title**: Adaptive Multi-LLM Orchestration Using Contextual Bandits
- **Subtitle**: Cost-Aware and Resource-Sensitive Dynamic Model Selection
- **Project Description**: An intelligent multi-LLM orchestration framework that dynamically routes user requests across local open-source models and cloud API providers based on prompt complexity, system resource state, model capabilities, and cost constraints. The system incorporates a 5-factor heuristic policy (`BaselineAdaptivePolicy`), a 1024D semantic embedding complexity classifier (`BAAI/bge-m3`), an offline Contextual Bandit reinforcement learning policy (`LinUCB`), and a Complex Task Decomposition Engine with deterministic structured synthesis ($S_2$).
- **Domain**: Machine Learning, Natural Language Processing, Reinforcement Learning, Distributed Systems, Software Engineering.
- **Problem Statement**: Single-model deployments enforce a rigid trade-off between inference cost, response latency, and capability. Large language models (e.g., DeepSeek R1 7B, Gemini 2.5 Pro) deliver high reasoning quality but incur high latency and cost, whereas smaller models (e.g., Gemma 3 4B, Qwen Coder 3B) are ultra-fast and low-cost but fail on complex reasoning. Fixed routing rules fail to adapt to dynamic system loads, fluctuating API availability, and nuanced query complexity. Furthermore, multi-stage task decomposition often creates severe downstream LLM latency bottlenecks during synthesis.
- **Research Problem**: How can an automated orchestrator dynamically select the optimal LLM (or execute a parallel DAG decomposition) for arbitrary queries to minimize latency and cost while preserving output quality, without relying on unvalidated real-time RL control?
- **Research Objectives**:
  1. Build a multi-provider routing engine supporting local Ollama models and cloud APIs (Google Gemini, Mistral, Groq).
  2. Develop a 1024D dense embedding classifier to quantify prompt complexity into 4 discrete tiers.
  3. Implement a multi-attribute utility policy (`BaselineAdaptivePolicy`) combining complexity, capability, latency, cost, and system resources.
  4. Formulate and shadow-evaluate an offline Contextual Bandit RL policy (`LinUCB`) with 12D state vectors and Inverse Propensity Scoring (IPS/SNIPS).
  5. Architect a Complex Task Decomposition Engine with parallel DAG execution and evaluate synthesis strategies ($S_0, S_1, S_2, S_3$) to eliminate synthesis bottlenecks.
  6. Rigorously validate the framework across 4 frozen empirical benchmark runs (`RUN_1` to `RUN_4`).
- **Motivation**: Maximize operational efficiency and quality-per-second in heterogeneous LLM environments while enforcing strict safety invariants (`production_override = False`).
- **Scope**: Covers prompt classification, single-model routing, multi-model DAG decomposition, parallel task scheduling, response validation, deterministic structured synthesis, offline RL policy evaluation, and backend/frontend engineering.

---

## 2. Team / Academic Information

- **University / Institution**: Amrita Vishwa Vidyapeetham
- **Department**: Department of Computer Science & Engineering
- **Academic Year**: 2025–2026
- **Course Context**: Reinforcement Learning (RL) Capstone / Final Year Project (Semester 7)
- **Student Names**: `NOT FOUND IN REPOSITORY`
- **Student Roll Numbers**: `NOT FOUND IN REPOSITORY`
- **Faculty Advisor / Guide**: `NOT FOUND IN REPOSITORY`

---

## 3. Abstract Facts

- **Objective**: Develop and evaluate an adaptive multi-LLM orchestration system that dynamically balances latency, quality, resource utilization, and cost using contextual routing and deterministic task synthesis.
- **Methodology**: 
  - 5-factor scoring model ($60\%$ Capability + Complexity match, $15\%$ Resource availability, $15\%$ Context length, $10\%$ Cost score).
  - 1024D BGE-M3 semantic embeddings with prototype cosine classification.
  - 12D state vector Contextual Bandit (LinUCB) running in offline shadow mode.
  - Complex Task Decomposition breaking prompts into parallel DAG levels.
  - Deterministic Structured Synthesis ($S_2$) aggregating subtask outputs without additional LLM overhead.
- **Key Empirical Results**:
  - **Single-Model Routing (`RUN_2` Local-Only V1, 791 runs)**: Strategy F (Capability Heuristic) and Strategy G (`BaselineAdaptivePolicy`) achieved $4.57/5.00$ and $4.58/5.00$ overall quality in $24.21\text{s}$ and $24.50\text{s}$ average latency, matching fixed DeepSeek R1 7B quality ($4.62/5.00$) while reducing latency by **$63.2\%$** ($65.78\text{s} \rightarrow 24.21\text{s}$).
  - **Task Decomposition (`RUN_3`, 36 runs)**: Complex task decomposition achieved a **$2.82\times$ subtask speedup** ($89.0\%$ parallel efficiency) over sequential execution, but revealed a critical synthesis bottleneck ($156.81\text{s}$ synthesis latency, accounting for $78.7\%$ of E2E latency).
  - **Synthesis Optimization (`RUN_4`, 72 runs)**: Replacing LLM synthesis with Deterministic Structured Assembly ($S_2$) reduced synthesis latency from $147.04\text{s}$ ($S_0$) to **$0.0008\text{s}$** ($S_2$), driving end-to-end latency down from $268.76\text{s}$ to **$121.72\text{s}$** (**$54.71\%$ latency reduction**, **$2.21\times$ measured E2E speedup**) with **$100\%$ objective coverage**.
  - **Engineering Status**: 215 of 216 pytest backend tests passed (1 manual test skipped as intended); Vite frontend production build compiled in $21.70\text{s}$.

---

## 4. Problem Statement

Modern AI deployments face a fundamental efficiency trilemma between **quality**, **latency**, and **cost**:

```text
               High Quality
              /            \
             /              \
    DeepSeek R1 / Gemini     Gemma 3 / Qwen Coder
           /                  \
          /                    \
   High Latency / Cost ----- Low Latency / Cost
```

1. **Model Diversity vs. Static Selection**: Deploying a single high-parameter model for all prompts wastes computation on trivial queries (e.g., greetings, basic math). Conversely, deploying a lightweight model causes failure on complex code generation and multi-step reasoning.
2. **Dynamic Hardware Constraints**: Local execution environments experience shifting CPU, RAM, and VRAM availability depending on concurrent workloads. Cloud APIs suffer from rate limits, network jitter, and service outages.
3. **Synthesis Bottlenecks in Decomposition**: While breaking complex queries into parallel subtasks improves subtask execution speed, conventional systems route all subtasks through a master LLM for final synthesis. This second-stage LLM step reinstates high latency and introduces summary truncation.

---

## 5. Objectives

1. **Unified Multi-Provider Integration**: Interface local Ollama models (`gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b`) and cloud APIs (Google Gemini, Mistral, Groq) under a seamless abstract provider interface.
2. **Semantic Complexity Classification**: Quantify prompt difficulty using a 1024D dense embedding model (`BAAI/bge-m3`) across 4 complexity tiers: `SIMPLE`, `MODERATE`, `COMPLEX`, `VERY_COMPLEX`.
3. **Adaptive Multi-Attribute Policy**: Execute routing decisions via `BaselineAdaptivePolicy` balancing capability match, complexity match, resource pressure, context length, and monetary cost.
4. **Contextual Bandit Shadow RL**: Train an offline LinUCB bandit policy using a 12D system state vector while maintaining strict production isolation (`production_override = False`).
5. **Parallel DAG Task Decomposition**: Dynamically split `COMPLEX` and `VERY_COMPLEX` prompts into structured subtasks with level-based asynchronous execution (`asyncio.gather()`).
6. **Deterministic Synthesis ($S_2$)**: Eliminate LLM synthesis overhead by implementing byte-reproducible structured aggregation.
7. **Empirical Benchmarking & Scientific Rigor**: Execute and freeze 4 major benchmark runs (`RUN_1` to `RUN_4`) with full SHA-256 telemetry tracking.

---

## 6. Key Contributions (mapped to source files)

| Contribution | Description | Source File Mapping |
| :--- | :--- | :--- |
| **Multi-Provider Pipeline** | Production orchestration pipeline supporting local and cloud LLM execution. | [`orchestrator/pipeline.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/orchestrator/pipeline.py), [`providers/factory.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/providers/factory.py) |
| **Baseline Adaptive Policy** | Production-authoritative routing heuristic based on 5 multi-attribute utility scores. | [`orchestrator/adaptive_policy.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/orchestrator/adaptive_policy.py) |
| **Complexity Classifier** | 1024D semantic embedding classification utilizing `BAAI/bge-m3`. | [`orchestrator/complexity_classifier.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/orchestrator/complexity_classifier.py) |
| **Shadow Contextual Bandit** | 12D state vector LinUCB RL policy with propensity logging & IPS/SNIPS evaluation. | [`rl/contextual_bandit.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/rl/contextual_bandit.py), [`rl/experience_buffer.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/rl/experience_buffer.py) |
| **Task Decomposition Engine** | Structured prompt parsing into parallel dependency DAGs. | [`orchestrator/task_decomposer.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/orchestrator/task_decomposer.py), [`orchestrator/parallel_scheduler.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/orchestrator/parallel_scheduler.py) |
| **Deterministic Synthesis ($S_2$)**| Zero-LLM-overhead structured assembly eliminating synthesis bottlenecks. | [`orchestrator/task_aggregator.py`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/orchestrator/task_aggregator.py) |
| **Validation Benchmark Suite**| Automated execution scripts, data processors, and statistical report generators. | [`validation/scripts/`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/validation/scripts/), [`validation/reports/`](file:///C:/Users/sachi/Desktop/Amrita/Sem-7/RL/Project/adaptive-llm-orchestrator/validation/reports/) |

---

## 7. System Architecture (A–I layers mapped to files)

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer A: User / API Interface (FastAPI REST Endpoints / Vite React Dashboard)           │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│ Layer B: Request Parsing & Context Extraction (orchestrator/pipeline.py)               │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│ Layer C: Semantic Vector & Complexity Classifier (orchestrator/complexity_classifier.py)│
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│ Layer D: Decision & Routing Engine (orchestrator/adaptive_policy.py & decision_engine) │
└──────────────────────────────────┬──────────────────────────────────┬──────────────────┘
                                   │                                  │
┌──────────────────────────────────▼─────────────┐ ┌──────────────────▼──────────────────┐
│ Layer E: RL Shadow Policy (LinUCB 12D State)   │ │ Layer G: Complex Task Decomposition & │
│ (rl/contextual_bandit.py - Shadow Only)        │ │ Parallel DAG Engine (task_decomposer) │
└────────────────────────────────────────────────┘ └──────────────────┬──────────────────┘
                                                                      │
┌─────────────────────────────────────────────────────────────────────▼──────────────────┐
│ Layer F: Provider Integration & Failover Layer (providers/ollama, gemini, mistral)      │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│ Layer H: Task Validation & Synthesis Layer (orchestrator/task_aggregator.py - S2)     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│ Layer I: Telemetry, Logging & Database Layer (database/connection.py, validation/raw)  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Layer A (Interface Layer)**: `main.py`, `frontend/src/`
- **Layer B (Request Parsing Layer)**: `orchestrator/pipeline.py`
- **Layer C (Complexity Classifier Layer)**: `orchestrator/complexity_classifier.py`
- **Layer D (Decision & Routing Layer)**: `orchestrator/adaptive_policy.py`, `orchestrator/decision_engine.py`
- **Layer E (RL Shadow Policy Layer)**: `rl/contextual_bandit.py`, `rl/experience_buffer.py`
- **Layer F (Provider & Failover Layer)**: `providers/factory.py`, `providers/base.py`, `providers/ollama_provider.py`
- **Layer G (Decomposition & DAG Layer)**: `orchestrator/task_decomposer.py`, `orchestrator/parallel_scheduler.py`
- **Layer H (Validation & Synthesis Layer)**: `orchestrator/task_validator.py`, `orchestrator/task_aggregator.py`
- **Layer I (Telemetry & Persistence Layer)**: `database/connection.py`, `database/models.py`, `validation/results/`

---

## 8. End-to-End Pipeline & Algorithmic Workflow

### 8.1 Single-Request Pipeline Lifecycle
1. **User Query Ingestion**: Client submits query via HTTP POST (`/api/v1/orchestrate`).
2. **Context & Hardware State Capture**: Read CPU load, RAM availability, GPU VRAM, active Ollama slots.
3. **Complexity Classification**: Query string is embedded into a 1024D dense vector via `BAAI/bge-m3`. Cosine distance to reference tier centroids assigns a complexity tier (`SIMPLE`, `MODERATE`, `COMPLEX`, `VERY_COMPLEX`).
4. **Policy Utility Calculation**: `BaselineAdaptivePolicy` computes utility score $U(m)$ for each registered model:
   $$U(m) = 0.40 \cdot \text{Match}(m) + 0.20 \cdot \text{Complexity}(m) + 0.15 \cdot \text{Resource}(m) + 0.15 \cdot \text{Context}(m) + 0.10 \cdot \text{Cost}(m)$$
5. **Model Selection & Failover Assignment**: Highest-scoring available model selected; backup candidate queue established.
6. **Shadow RL Logging**: 12D state vector created; LinUCB policy predicts action $a_{RL}$ in shadow mode without overriding selection (`production_override = False`).
7. **Model Execution**: Request dispatched to selected provider (e.g., local Ollama instance).
8. **Response Return & Telemetry Update**: Output returned to user; latency, token count, and resource state persisted to SQLite database.

### 8.2 Complex Task Decomposition Lifecycle
1. **Trigger Condition**: Query classified as `COMPLEX` or `VERY_COMPLEX` and decomposition flag enabled.
2. **JSON DAG Generation**: `DynamicTaskDecomposer` queries model to break user prompt into subtasks with explicit dependencies (`depends_on`).
3. **DAG Scheduling**: `ParallelTaskScheduler` categorizes subtasks into sequential execution levels.
4. **Concurrent Subtask Execution**: Independent subtasks in Level $k$ execute asynchronously via `asyncio.gather()`.
5. **Response Validation**: `TaskResponseValidator` inspects subtask outputs for non-empty completion and relevance.
6. **Deterministic Synthesis ($S_2$)**: `TaskAggregator` formats subtask outputs into structured markdown sections (`# Executive Summary`, `# Subtask Outputs`, `# Technical Details`). Zero LLM calls executed during synthesis.

---

## 9. Complexity Classifier & State Vector Specification

### 9.1 Semantic Embedding Complexity Classifier
- **Embedding Model**: `BAAI/bge-m3` (1024D dense vector representation).
- **Heuristic Feature Extractor**:
  - Prompt character/token length.
  - Code snippet token density (detecting `{`, `}`, `def`, `class`, `import`, `return`, `function`).
  - Technical keyword density (algorithmic, math, and system terms).
  - Structural formatting depth (markdown headers, bullet lists, nested indentations).
  - Multi-step reasoning requirements (presence of "explain step by step", "analyze", "compare").
- **Complexity Tiers & Thresholds**:
  - `SIMPLE` ($[0.00, 0.30)$): Basic Q&A, greetings, definitions.
  - `MODERATE` ($[0.30, 0.55)$): Short code snippets, multi-sentence summaries.
  - `COMPLEX` ($[0.55, 0.80)$): Multi-file code generation, detailed architecture design.
  - `VERY_COMPLEX` ($[0.80, 1.00]$): Advanced multi-objective decomposition tasks.

### 9.2 12D Context State Vector Specification

| Index | Feature Name | Range | Description / Normalization |
| :---: | :--- | :---: | :--- |
| `0` | Prompt Token Length | $[0, 1]$ | $\min(1.0, \text{tokens} / 4000)$ |
| `1` | Code Token Density | $[0, 1]$ | Ratio of code syntax tokens to total tokens |
| `2` | Technical Keyword Density | $[0, 1]$ | Frequency of domain-specific technical terms |
| `3` | Structural Depth Ratio | $[0, 1]$ | Markdown header and list structure score |
| `4` | Calculated Complexity Score | $[0, 1]$ | Output score from `ComplexityClassifier` |
| `5` | Peak Hour Indicator | $\{0, 1\}$ | $1.0$ if current time falls within peak hours, else $0.0$ |
| `6` | System CPU Utilization | $[0, 1]$ | Host CPU load ratio (`psutil.cpu_percent() / 100`) |
| `7` | System RAM Utilization | $[0, 1]$ | Host RAM load ratio (`psutil.virtual_memory().percent / 100`) |
| `8` | GPU VRAM Utilization | $[0, 1]$ | VRAM load ratio (via `pynvml` or Ollama management) |
| `9` | Active Ollama Slots | $[0, 1]$ | Concurrent active inference jobs ratio |
| `10` | Provider Health Index | $[0, 1]$ | $1.0 - \text{recent API error rate}$ |
| `11` | Context Depth Ratio | $[0, 1]$ | Historical conversation context length ratio |

---

## 10. Contextual Bandit RL Formulation & Shadow Mode

### 10.1 Reinforcement Learning Formulation
- **Algorithm**: Linear Upper Confidence Bound (LinUCB) Contextual Bandit.
- **Action Space**: $A = \{0, 1, 2, 3, 4, 5, 6, 7\}$ corresponding to model registry actions.
- **Action 7 Masking**: Action 7 (`BAAI/bge-m3`) is strictly **masked from LLM generation** during bandit action selection, as BGE-M3 is reserved exclusively for 1024D vector embeddings.
- **Reward Function**: Multi-objective utility combining response quality score $Q \in [1, 5]$, normalized latency $L \in [0, 1]$, and cost $C \in [0, 1]$:
  $$R = w_q \cdot \left(\frac{Q}{5}\right) - w_l \cdot L - w_c \cdot C$$

### 10.2 Production Isolation Invariant
- **Hardcoded Isolation**: `production_override = False` in `AdaptiveDecisionEngine`.
- **Policy Role**: The RL agent operates strictly in **shadow mode**. It receives state vectors $s_t$, logs predicted optimal action $a_{RL}$ and propensity $p(a_{RL}|s_t)$, but **never overrides** `BaselineAdaptivePolicy`.
- **Offline Evaluation**: Evaluated using Inverse Propensity Scoring (IPS) and Self-Normalized Inverse Propensity Scoring (SNIPS) over recorded experience logs (`ExperienceBufferService`).

---

## 11. Models & Providers Inventory Table

| Index | Model ID / Name | Provider | Endpoint / Type | Param Size | Primary Assigned Role |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **0** | `gemma-3-4b` | Ollama | Local (`localhost:11434`) | 4B | Simple Q&A, ultra-fast routing target |
| **1** | `qwen-coder-3b` | Ollama | Local (`localhost:11434`) | 3B | Fast code generation & lightweight logic |
| **2** | `deepseek-r1-7b` | Ollama | Local (`localhost:11434`) | 7B | Deep reasoning, complex logic decomposition |
| **3** | `gemini-2.5-flash` | Google Gemini | Cloud API | API | Medium-cost fast cloud fallback |
| **4** | `gemini-2.5-pro` | Google Gemini | Cloud API | API | High-reasoning complex cloud fallback |
| **5** | `mistral-small` | Mistral AI / Groq | Cloud API | API | General natural language cloud processing |
| **6** | `mistral-large` | Mistral AI / Groq | Cloud API | API | High-capability multi-lingual cloud fallback |
| **7** | `BAAI/bge-m3` | Local HuggingFace | Local Memory | 560M | **Masked from generation**; 1024D embeddings |

---

## 12. Provider Failover Mechanism

1. **Exhausted Provider Tracking**: Maintains a transient failure set $F$ during request dispatch.
2. **Candidate Ranking**: `BaselineAdaptivePolicy` ranks all active models by utility score $U(m)$.
3. **Primary Dispatch**: Attempt dispatch to rank-1 model $m_1 \notin F$.
4. **Retry & Backoff**: If provider returns connection error, rate limit (HTTP 429), or timeout, $m_1$ added to $F$.
5. **Fallback Execution**: Dynamic failover automatically re-routes request to next highest utility model $m_2 \notin F$.
6. **Circuit Breaker Integration**: Providers with consecutive error counts exceeding threshold $\tau_{fail}$ marked temporarily inactive for cooldown window $T_{cool} = 60\text{s}$.

---

## 13. Complex Task Decomposition Engine

- **Architecture**: `DynamicTaskDecomposer` coupled with `ParallelTaskScheduler`.
- **DAG Schema Specification**:
  ```json
  {
    "subtasks": [
      {
        "id": "task_1",
        "title": "System Architecture Overview",
        "description": "Design core modular architecture...",
        "depends_on": []
      },
      {
        "id": "task_2",
        "title": "Database Schema Definition",
        "description": "Define relational schema...",
        "depends_on": ["task_1"]
      }
    ]
  }
  ```
- **Parallel Level Scheduler**: Subtasks categorized into topological levels $L_0, L_1, \dots, L_k$.
- **Asynchronous Execution**: Subtasks in level $L_j$ execute concurrently via Python `asyncio.gather()`.
- **Resource Constraints**: Concurrency capped by available host CPU threads and local Ollama slot capacity.

---

## 14. Deterministic Synthesis Strategies

| Strategy | Name | LLM API Calls | Synthesis Latency | Summary Quality | Reproducibility |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$S_0$** | Direct LLM Synthesis (DeepSeek R1 7B) | 1 | $147.04\text{s}$ | High | Non-deterministic |
| **$S_1$** | Lightweight LLM Synthesis (Gemma 4B) | 1 | $45.03\text{s}$ | Moderate | Non-deterministic |
| **$S_2$** | **Deterministic Structured Assembly (DEFAULT)** | **0** | **$0.0008\text{s}$** | **$100\%$ Coverage** | **$100\%$ Exact Byte Match** |
| **$S_3$** | Hybrid Assembly + Gemma Summary | 1 | $4.51\text{s}$ | Concise Summary | Non-deterministic |

- **$S_2$ Implementation Mechanism**: `TaskAggregator` parses completed subtask outputs, validates response non-emptiness via `TaskResponseValidator`, and concats structured Markdown sections without invoking LLM generation. Eliminates $100\%$ of LLM synthesis overhead.

---

## 15. Validation Framework Scope & Suite

- **Dataset**: `validation/datasets/prompts.json` (113 curated multi-domain benchmark prompts).
- **Automated Runner**: `validation/scripts/run_benchmark.py`.
- **Evaluation Criteria**: Objective Coverage ($0-100\%$), Response Quality ($1.0-5.0$), E2E Latency (seconds), Generation Speed (tokens/sec), monetary cost (\$ USD).
- **Reproducibility Controls**: Raw telemetry saved as uncompressed JSONL with full timestamping and SHA-256 manifest verification.

---

## 16. RUN_1 Results & Artifacts

- **Scope**: Multi-provider live environment evaluation (113 prompts).
- **Directory**: `validation/results/raw/`
- **Key Findings**: Live cloud API rate limiting (HTTP 429) introduced high variance in latency measurements. Prompt complexity classification was validated, but noisy network environments highlighted the need for a 100% controlled local benchmark (`RUN_2`).

---

## 17. RUN_2 Results (LOCAL_ONLY_V1)

- **Scope**: 113 benchmark prompts $\times$ 7 strategies (A–G) = **791 executions**.
- **Execution Environment**: 100% Local Ollama, 100% execution success, 0 cloud API calls.
- **Directory**: `validation/results/local_only/`

### Strategy Performance Summary Table (`RUN_2`)

| Strategy ID | Strategy Name | Mean Quality (1-5) | Mean Latency (s) | E2E Speedup vs DeepSeek | Success Rate |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **A** | Fixed Gemma 3 4B | $4.12$ | $12.45\text{s}$ | $5.28\times$ | $100\%$ |
| **B** | Fixed Qwen Coder 3B | $4.25$ | $15.10\text{s}$ | $4.36\times$ | $100\%$ |
| **C** | Fixed DeepSeek R1 7B | $4.62$ | $65.78\text{s}$ | $1.00\times$ | $100\%$ |
| **D** | Random Selection | $4.31$ | $31.15\text{s}$ | $2.11\times$ | $100\%$ |
| **E** | Round-Robin | $4.33$ | $30.88\text{s}$ | $2.13\times$ | $100\%$ |
| **F** | Capability Heuristic | $4.57$ | $24.21\text{s}$ | $2.72\times$ | $100\%$ |
| **G** | **BaselineAdaptivePolicy** | **$4.58$** | **$24.50\text{s}$** | **$2.68\times$** | **$100\%$** |

- **Key Takeaway**: Strategy F and G matched fixed DeepSeek R1 7B quality ($4.58$ vs $4.62$, $p > 0.05$) while reducing average latency by **$63.2\%$** ($65.78\text{s} \rightarrow 24.21\text{s}$).

---

## 18. RUN_3 Results (Corrected Metric Audit)

- **Scope**: $N = 18$ complex prompts, 36 real executions comparing Direct Single-Model (DeepSeek R1 7B) vs. Complex Task Decomposition Pipeline.
- **Directory**: `validation/results/complex_task_comparison/`

### Direct vs. Orchestrated Comparison Table (`RUN_3`)

| Metric Domain | Direct Single Model (DeepSeek R1 7B) | Complex Task Orchestrated Pipeline | Delta / Speedup Ratio |
| :--- | :---: | :---: | :---: |
| **Overall Quality Score** | $4.37 / 5.00$ | $4.25 / 5.00$ | $-0.12$ ($p = 0.412$, Statistically Equivalent) |
| **Objective Coverage** | $89.02\%$ | $85.02\%$ | $-4.00\%$ ($p = 0.389$, Statistically Equivalent) |
| **Subtask Latency (Sequential)** | N/A | $562.14\text{s}$ | N/A |
| **Subtask Latency (Parallel)** | N/A | $199.34\text{s}$ | **$2.82\times$ Parallel Subtask Speedup** |
| **Parallel Efficiency** | N/A | $89.0\%$ | High DAG Parallel Efficiency |
| **Synthesis Phase Latency** | $0.00\text{s}$ | $156.81\text{s}$ | **$78.7\%$ of Total E2E Latency Bottleneck** |

---

## 19. RUN_4 Results (Measured Synthesis Ablation)

- **Scope**: $N = 18$ complex prompts across 4 synthesis modes ($S_0, S_1, S_2, S_3$) = **72 trials**.
- **Directory**: `validation/results/complex_task_synthesis_optimization/`

### Measured Synthesis Optimization Results Table (`RUN_4`)

| Synthesis Strategy Mode | Synthesis Latency (s) | Total E2E Latency (s) | E2E Latency Reduction vs $S_0$ | Measured E2E Speedup vs $S_0$ | Objective Coverage | Byte Reproducibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S_0$ (DeepSeek 7B LLM)** | $147.04\text{s}$ | $268.76\text{s}$ | Baseline ($0.0\%$) | $1.00\times$ | $91.67\%$ | Non-deterministic |
| **$S_1$ (Gemma 4B LLM)** | $45.03\text{s}$ | $166.75\text{s}$ | $-37.96\%$ | $1.61\times$ | $91.67\%$ | Non-deterministic |
| **$S_2$ (Deterministic Default)**| **$0.0008\text{s}$** | **$121.72\text{s}$** | **$-54.71\%$** | **$2.21\times$** | **$100.0\%$** | **$100\%$ Byte Match** |
| **$S_3$ (Hybrid Summary)** | $4.51\text{s}$ | $126.23\text{s}$ | $-53.03\%$ | $2.13\times$ | $100.0\%$ | Non-deterministic |

- **Key Measured Findings**:
  - $S_2$ reduced synthesis latency from $147.04\text{s}$ to **$0.0008\text{s}$** ($99.999\%$ drop).
  - Total end-to-end pipeline latency dropped from $268.76\text{s}$ ($S_0$) to **$121.72\text{s}$** ($S_2$), representing a measured **$54.71\%$ E2E latency reduction** and **$2.21\times$ measured speedup**.
  - Objective coverage increased from $91.67\%$ ($S_0$) to **$100.0\%$** ($S_2$) by preserving all generated subtask outputs without LLM summary truncation.

---

## 20. Final Test & Repository Status

### 20.1 Test Suite Status
- **Backend Pytest Suite**: **215 Passed**, **1 Skipped** (`test_manual_live_call`), **0 Failed** (Execution time: $190.65\text{s}$).
- **Synthesis Unit Tests**: 1 Passed out of 1.
- **Parallel DAG Scheduler Tests**: 9 Passed out of 9.
- **Python Syntax (`compileall`)**: $100\%$ Clean (0 syntax errors).
- **Frontend Production Build (`npm run build`)**: **$100\%$ Success** (`built in 21.70s`).

---

## 21. Security, Safety, & Environment Configuration

- **Key Management**: All provider API keys (`GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`) loaded exclusively from `.env` via Pydantic `BaseSettings` (`config.py`).
- **Zero Hardcoded Secrets**: Scanned repository confirms zero plain-text API credentials.
- **Log Sanitation**: Telemetry logs record model IDs, latency numbers, and token counts while strictly redacting raw API keys or header authorization strings.
- **Production Safety Invariant**: `production_override = False` explicitly hardcoded in `AdaptiveDecisionEngine`.

---

## 22. Limitations & Threats to Validity

1. **Hardware Specificity**: Local Ollama executions evaluated on a single host setup (NVIDIA GPU / CPU host). Hardware resource variations will alter absolute latency numbers.
2. **LLM Evaluator Variance**: Quality scoring utilized LLM-as-a-judge evaluation. While standard prompts were used, evaluator model non-determinism remains a minor factor.
3. **Cloud API Rate Limits**: `RUN_1` live API tests suffered from free-tier rate limits, necessitating the controlled environment of `RUN_2`.
4. **Decomposition Overhead**: For simple or moderate prompts, task decomposition introduces net latency overhead; decomposition must be restricted strictly to `COMPLEX` and `VERY_COMPLEX` queries.

---

## 23. Future Work & Research Directions

1. **Online RL Deployment**: Gradually transition LinUCB Contextual Bandit from shadow mode to online production control using off-policy safety bounds.
2. **Multi-Modal Orchestration**: Extend routing policy to support vision-language models (e.g., Gemini 2.5 Flash Vision, LLaVA).
3. **Adaptive Subtask Pruning**: Implement dynamic DAG pruning to eliminate redundant subtasks during level execution based on intermediate validation feedback.

---

## 24. Figure Inventory Table

| Figure ID | File Path / Artifact Reference | Title / Description |
| :---: | :--- | :--- |
| **Fig. 1** | `validation/reports/figures/architecture.png` | System Layer Architecture Diagram |
| **Fig. 2** | `validation/reports/figures/complexity_distribution.png` | 1024D Embedding Complexity Tier Distribution |
| **Fig. 3** | `validation/reports/figures/run2_quality_vs_latency.png` | `RUN_2` Quality vs. Latency Strategy Pareto Frontier |
| **Fig. 4** | `validation/reports/figures/dag_execution_timeline.png` | Parallel DAG Level Execution Timeline |
| **Fig. 5** | `validation/reports/figures/run4_synthesis_ablation.png` | `RUN_4` Synthesis Mode Latency & Coverage Ablation ($S_0 \rightarrow S_3$) |
| **Fig. 6** | `presentation.md` (Slide 4) | Routing Decision Matrix Flow |

---

## 25. Table Inventory Table

| Table ID | Report Reference | Title / Subject |
| :---: | :--- | :--- |
| **Tab. 1** | Report 33 (Sec. 9.2) | 12D Context State Vector Specification |
| **Tab. 2** | Report 33 (Sec. 11) | Models & Providers Inventory Matrix |
| **Tab. 3** | Report 26 / Report 33 (Sec. 17) | `RUN_2` Local-Only Strategy Performance Comparison |
| **Tab. 4** | Report 27 / Report 33 (Sec. 18) | `RUN_3` Direct vs. Orchestrated Decomposition Audit |
| **Tab. 5** | Report 29 / Report 33 (Sec. 19) | `RUN_4` Synthesis Optimization Ablation Results |
| **Tab. 6** | Report 32 / Report 33 (Sec. 20) | Comprehensive Pytest & Build Status Matrix |

---

## 26. References / Citations in Repository

1. **Li, L., Chu, W., Langford, J., & Schapire, R. E.** (2010). *A contextual-bandit approach to personalized news article recommendation*. Proceedings of the 19th international conference on World Wide Web (WWW).
2. **Dudík, M., Langford, J., & Li, L.** (2011). *Doubly robust policy evaluation and learning*. arXiv preprint arXiv:1103.4601.
3. **BAAI Team** (2023). *BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings*. Beijing Academy of Artificial Intelligence.
4. **Zheng, L., et al.** (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.
5. **Ollama Project** (2024). *Get up and running with Llama 3, Mistral, Gemma, and other large language models*. `https://ollama.com`

---

## 27. Implementation File Map

```text
adaptive-llm-orchestrator/
├── main.py                               # FastAPI application entry point & routes
├── config.py                             # System configuration & environment settings
├── orchestrator/
│   ├── pipeline.py                       # Main Orchestration Pipeline execution engine
│   ├── adaptive_policy.py                # BaselineAdaptivePolicy (5-factor heuristic routing)
│   ├── decision_engine.py                # AdaptiveDecisionEngine (Production vs Shadow routing)
│   ├── complexity_classifier.py          # BGE-M3 1024D embedding prompt complexity classifier
│   ├── task_decomposer.py                # DynamicTaskDecomposer (JSON DAG generator)
│   ├── parallel_scheduler.py             # ParallelTaskScheduler (asyncio DAG level executor)
│   ├── task_validator.py                 # TaskResponseValidator (subtask quality inspector)
│   └── task_aggregator.py                # TaskAggregator (S0, S1, S2, S3 synthesis strategies)
├── providers/
│   ├── base.py                           # Abstract Base Provider interface class
│   ├── factory.py                        # Provider Factory instantiation engine
│   ├── ollama_provider.py                # Local Ollama Provider implementation
│   ├── gemini_provider.py                # Cloud Google Gemini Provider implementation
│   ├── mistral_provider.py               # Cloud Mistral AI Provider implementation
│   └── groq_provider.py                  # Cloud Groq Provider implementation
├── rl/
│   ├── contextual_bandit.py              # LinUCB Contextual Bandit RL policy
│   ├── experience_buffer.py              # Experience Replay Buffer & telemetry logger
│   └── offline_evaluator.py              # IPS / SNIPS offline policy evaluator
├── database/
│   ├── connection.py                     # SQLite database connection setup
│   └── models.py                         # SQLAlchemy database tables & schemas
├── frontend/
│   ├── src/                              # Vite React Frontend Dashboard code
│   └── package.json                      # Frontend dependencies & scripts
├── validation/
│   ├── datasets/prompts.json             # 113 benchmark prompts dataset
│   ├── scripts/                          # Automated benchmark runner scripts
│   ├── results/                          # Frozen raw telemetry data (RUN_1 to RUN_4)
│   └── reports/                          # Scientific reports (01_*.md to 33_*.md)
└── tests/                                # Pytest test suite (215 passing test cases)
```

---
**End of Fact Sheet — Ready for Overleaf IEEE Report Compilation.**
