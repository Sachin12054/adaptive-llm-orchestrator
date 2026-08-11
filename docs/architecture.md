# System Architecture — Adaptive AI Orchestration Platform (Phase 1)

## Current Implemented Workflow (Complex Task Allocation Layer)

```
                                 USER PROMPT
                                      │
                               INPUT PROCESSING
                                      │
                             BGE-M3 EMBEDDING
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      INTENT CLASSIFIER       COMPLEXITY ANALYZER     RESOURCE TELEMETRY
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      ▼
                          ADAPTIVE DECISION ENGINE
                                      │
     ┌────────────────────────────────┼────────────────────────────────┐
     ▼                                ▼                                ▼
  SIMPLE                           MEDIUM                           COMPLEX
(gemma-3-4b)               (qwen-coder-3b / gemma)          (NEW TASK ALLOCATION)
     │                                │                                │
     ▼                                ▼                                ▼
Normal Single Execution        Normal Single Execution            DECOMPOSER
                                                                       │
                                                               SUBTASK IDENTIFIER
                                                                       │
                                                                DEPENDENCY GRAPH
                                                                       │
                                                                TASK ALLOCATION
                                                                       │
                                                     ┌─────────────────┼─────────────────┐
                                                     ▼                 ▼                 ▼
                                                   GEMMA           DEEPSEEK            QWEN
                                                  GENERAL          REASONING          CODING
                                                     │                 │                 │
                                                     └─────────────────┼─────────────────┘
                                                                       ▼
                                                             TASK PLAN OUTPUT ONLY
                                                           (Subtasks not executed yet)
```

---

## Detailed Dataflow & Pipeline Layers

```
USER PROMPT
   │
   ▼
INPUT PROCESSING (backend/app/services/input_processor.py)
   │
   ▼
BGE-M3 EMBEDDING (backend/app/services/embedding_service.py)
   │
   ▼
SEMANTIC REPRESENTATION (1024-dimensional dense vector)
   │
   ├──────────────────────────┬──────────────────────────┬──────────────────────────┐
   ▼                          ▼                          ▼                          ▼
INTENT CLASSIFIER      COMPLEXITY ANALYZER      RESOURCE TELEMETRY         MODEL REGISTRY
(Step 9)               (Step 10)                (Step 11)                  (Step 12)
   │                          │                          │                          │
   └──────────────────────────┼──────────────────────────┴──────────────────────────┘
                              ▼
                  ADAPTIVE DECISION ENGINE (Step 15)
                              │
                              ├─ Active Production Routing: BaselineAdaptivePolicy
                              └─ Shadow Evaluation Mode: RLContextualBanditPolicy (Step 21 Shadow Proposals)
                              │
                              ▼
                 Decision Classification & Model Allocation
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   SIMPLE / MEDIUM PROMPTS             COMPLEX PROMPTS (complexity >= 0.80)
               │                             │
               ▼                             ▼
       Single Execution Path             COMPLEX TASK ALLOCATION LAYER
               │                         (backend/app/services/complex/)
               │                             │
               │                             ├─ ComplexTaskDecomposer
               │                             ├─ DependencyGraphBuilder (DAG & Execution Levels)
               │                             └─ TaskAllocator (Maps to 3 Local Ollama Models)
               │                             │
               ▼                             ▼
        MODEL MANAGER                 RETURN TASK PLAN ONLY
               │                  (Subtasks not executed in this milestone)
       OllamaProvider
   (http://localhost:11434)
               │
               ▼
     Response Verification (Step 17)
               │
               ▼
     Deterministic Reward (Step 18)
               │
               ▼
  Experience Replay Buffer (Step 20)
               │
               ▼
    Offline RL Shadow Eval (Step 21)
```

---

## Stage Responsibilities & Architecture

### 1. Input Processing
- **Sanitization & Normalization**: Strips extraneous whitespace and consolidates internal tokens (`backend/app/services/input_processor.py`).
- **Validation**: Enforces strict non-empty and non-whitespace check and `MAX_PROMPT_LENGTH` limits (default: 10,000 characters).

### 2. Semantic Embedding (BGE-M3)
- **Dense Vector Encoding**: Uses `BAAI/bge-m3` via `sentence-transformers` to produce a 1024-dimensional dense vector representation.

### 3. Semantic Intent Classification
- **Cosine Similarity Alignment**: Evaluates high-dimensional vector similarity against prototype vectors across intent classes (`factual`, `coding`, `mathematics`, `reasoning`, `summarization`, `explanation`, `creative`, `translation`, `conversational`).

### 4. Complexity Analyzer
- **Multi-Factor Scoring**: Combines 5 normalized factors: `semantic_complexity` (0.30), `reasoning_complexity` (0.25), `task_complexity` (0.20), `context_complexity` (0.15), `output_complexity` (0.10).
- **Complexity Level Classification**: Maps score to `low` (<0.30), `medium` (<0.60), `high` (<0.80), or `very_high` (>=0.80).

### 5. Complex Task Allocation Layer (`backend/app/services/complex/`)
- **ComplexTaskDecomposer**: Evaluates whether a prompt is complex. Breaks down complex prompts into structured subtasks with unique task IDs (`TASK-1`, `TASK-2`, ...), descriptions, categories, and dependency constraints.
- **DependencyGraphBuilder**: Constructs a Directed Acyclic Graph (DAG) for subtasks, performs cycle detection via DFS, and computes `execution_levels` representing parallel execution groups.
- **TaskAllocator**: Deterministically maps subtask categories to local Ollama models:
  - `coding` $\rightarrow$ `qwen-coder-3b`
  - `reasoning` / `mathematics` / `analysis` $\rightarrow$ `deepseek-r1-7b`
  - `general` / `explanation` / `factual` / `conversational` / `summarization` $\rightarrow$ `gemma-3-4b`
  - Ambiguous / unknown $\rightarrow$ `gemma-3-4b`
- **Milestone Guarantee**: Returns the `ComplexTaskPlan` **ONLY**. Subtasks are NOT executed through ModelManager in this milestone.

### 6. Model Registry & Local Providers
- **Structured Model Registry**: Manages declarations for local Ollama executable models (`gemma-3-4b`, `qwen-coder-3b`, `deepseek-r1-7b`) and BGE-M3 embedding.
- **OllamaProvider Adapter**: Connects to `http://localhost:11434/api/generate` without cloud API dependencies.

### 7. Reward Signal & Experience Buffer (Steps 18 & 20)
- **Deterministic Reward Signal**: Computes scalar reward $[0.0 - 1.0]$ based on structural quality (0.30), completeness (0.20), relevance (0.25), verification pass (0.15), and execution success (0.10).
- **Experience Replay Buffer**: Stores transition records with **12-dimensional state vector** and **exact Step 18 reward**.

### 8. Offline Contextual Bandit Policy (Step 21)
- **3-Action Policy**: Action 0 $\rightarrow$ `gemma-3-4b`, Action 1 $\rightarrow$ `qwen-coder-3b`, Action 2 $\rightarrow$ `deepseek-r1-7b`.
- **Shadow Mode Operation**: Operates strictly in shadow evaluation mode without overriding `BaselineAdaptivePolicy` production routing.
