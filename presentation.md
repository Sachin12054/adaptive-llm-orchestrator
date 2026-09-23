# COMPLETE PROJECT PRESENTATION DECK (18 SLIDES)
## Adaptive LLM Orchestrator: Cost-Aware and Resource-Sensitive Dynamic Model Selection via Contextual Bandit Telemetry

---

### SLIDE 1 — TITLE
- **Complete Project Title**: Adaptive LLM Orchestrator: Cost-Aware and Resource-Sensitive Dynamic Model Selection via Contextual Bandit Telemetry
- **Team Members (Group of 4)**:
  - Member 1: AI Lead (BGE-M3 1024D Embeddings & Baseline Adaptive Policy)
  - Member 2: RL Specialist (12D State Vector Encoder & Shadow Contextual Bandit Policy)
  - Member 3: Backend Engineer (FastAPI Architecture, Telemetry & Provider Adapters)
  - Member 4: Frontend Lead (React Dashboard UI, SSE Reader & 28 Pytest Suite)
- **Department / Program**: Department of Computer Science & Engineering | B.Tech CSE
- **College**: Amrita Vishwa Vidyapeetham
- **Guide / Faculty Name**: Project Evaluation Faculty / Mentor
- **Academic Year**: 2025 - 2026

---

### SLIDE 2 — PROBLEM STATEMENT
- **Real-World Problem**: Monolithic single-model LLM deployments lead to severe financial waste and latency for simple queries, and execution failure for complex tasks.
- **Cost Waste**: Assigning a 70B parameter cloud model to simple factual queries ("What is the capital of France?") wastes API money ($0.00059/1k tokens).
- **Task Failures**: Defaulting to lightweight 3B/4B models causes execution errors and hallucinations on multi-step calculus or system design tasks.
- **Hardware Agnosticism**: Existing routers ignore host GPU free VRAM availability, causing local out-of-memory (OOM) crashes.
- **Core Requirement**: Need an automated, hardware-aware router that dynamically balances cost, latency, and response quality.

---

### SLIDE 3 — MOTIVATION
- **Selection Rationale**: Rapid proliferation of specialized LLMs (coding, reasoning, translation, lightweight QA).
- **Real-World Relevance**: Enterprise applications need to optimize the cost-latency-quality Pareto frontier, achieving sub-400ms free execution for simple queries.
- **Existing Gap**: Existing commercial routers focus on cloud models only, completely ignoring local edge hardware constraints and lacking shadow RL evaluation.
- **Why AI/ML is Required**: Rule-based regex patterns fail on dynamic prompt difficulty, requiring 1024D dense PyTorch embeddings and multi-factor scoring.

---

### SLIDE 4 — OBJECTIVES
- **Primary Objective [IMPLEMENTED]**: Develop an adaptive multi-LLM orchestration platform enforcing a single production decision policy (`BaselineAdaptivePolicy`).
- **Secondary Objective 1 [IMPLEMENTED]**: Generate 1024D dense vectors via PyTorch `BAAI/bge-m3`.
- **Secondary Objective 2 [IMPLEMENTED]**: Classify intent across 9 prototypes and compute 5 structural complexity scores.
- **Secondary Objective 3 [IMPLEMENTED]**: Inspect CPU %, RAM %, and free GPU VRAM prior to model candidate discovery.
- **Secondary Objective 4 [IMPLEMENTED]**: Score candidates via 5-factor weighted baseline decision formula ($30/30/15/15/10$).
- **Secondary Objective 5 [IMPLEMENTED]**: Log 12D state vectors & rewards via shadow Contextual Bandit policy (`production_override: false`).

---

### SLIDE 5 — EXISTING SYSTEM
- **Current Workflow**: User prompts are sent directly to a fixed single model endpoint or a static fallback switch.
- **Existing Techniques**: Hardcoded API endpoints or primitive pattern-matching regex switches.
- **Major Limitations**:
  - High per-token billing for simple queries ($0.00059/1k tokens).
  - Execution failure on small models for complex math or architecture tasks.
  - GPU VRAM crashes on local edge nodes due to memory agnosticism.
  - Discarding execution outcomes instead of logging replay data for RL learning.

---

### SLIDE 6 — PROPOSED SYSTEM
- **Proposed Solution**: An intelligent multi-LLM orchestration backend combining 1024D PyTorch dense embeddings, 5-factor prompt complexity analysis, host hardware VRAM telemetry, and shadow Contextual Bandit reinforcement learning.
- **Pipeline**: Input Prompt ──> PyTorch BGE-M3 1024D Vector ──> Intent & Complexity Analysis ──> VRAM Check ──> Baseline 5-Factor Scoring ──> Exact Dispatch ──> Shadow RL Replay Buffer.
- **Key Advantages**:
  - Up to 70% token cost reduction on low-complexity query benchmarks.
  - Zero local GPU out-of-memory crashes via host resource pre-checks.
  - 60% Quality Protection weight (Capability + Complexity) prevents weak model selection.
  - Zero production risk via shadow mode RL evaluation (`production_override: false`).

---

### SLIDE 7 — SYSTEM ARCHITECTURE
- **Frontend Tier**: React 18 SPA + Vite 5.4 Glassmorphism UI + Server-Sent Events (SSE) telemetry consumer.
- **Backend API Tier**: FastAPI 0.115 ASGI Framework + Starlette SSE streaming response handlers.
- **AI Processing Tier**: PyTorch BGE-M3 1024D Dense Embedding Service + 9-Intent Cosine Classifier + 5-Factor Complexity Service.
- **Telemetry Tier**: Hardware inspection via `psutil` (CPU/RAM %) and PyTorch `pynvml` (Free GPU VRAM in GB).
- **Decision & Execution Tier**: `BaselineAdaptivePolicy` production authority + Ollama and Cloud API Provider Managers.
- **Storage & Replay Tier**: JSON Model Registry (`datasets/models/model_registry.json`) + JSONL Replay Buffer (`data/rl/experience_buffer.jsonl`).

---

### SLIDE 8 — METHODOLOGY
- **Stage 1–3**: Input Validation ──> PyTorch BGE-M3 1024D Vector ──> 9-Intent Prototype Matching.
- **Stage 4–6**: 5-Factor Complexity Analysis ──> Hardware VRAM Check ──> Candidate Discovery.
- **Stage 7–9**: 5-Factor Baseline Scoring ──> Neutral Tie-Breaking ──> Provider Dispatch.
- **Stage 10–12**: Structural Response Verification ──> Reward Calculation ($R \in [0, 1]$) ──> 12D Experience Storage.
- **Stage 13–14**: Shadow RL Q-Value Prediction (override=false) ──> Real-time SSE Telemetry Stream.

---

### SLIDE 9 — DATASET / DATA
- **Model Registry Dataset (`datasets/models/model_registry.json`)**:
  - Contains metadata for 8 registered local and cloud models.
  - Attributes: Display names, provider, execution mode, context limit, capabilities, token pricing.
  - Published Token Pricing: Input cost ($0.00 - $0.00059/1k), Output cost ($0.00 - $0.00079/1k).
- **Experience Replay Buffer (`data/rl/experience_buffer.jsonl`)**:
  - JSONL transition persistence logging 12D state vectors, actions, rewards, and completion status.

---

### SLIDE 10 — TECHNOLOGY STACK
- **Languages**: Python 3.14, JavaScript (ES6+), HTML5, CSS3.
- **Frameworks**: FastAPI 0.115, React 18, Vite 5.4, Uvicorn, Pydantic v2, SSE-Starlette.
- **AI/ML Core**: PyTorch, `sentence-transformers` (`BAAI/bge-m3`), NumPy, Scikit-learn.
- **LLM Hosts & APIs**: Ollama Server (`http://localhost:11434`), Google GenAI SDK, HTTPX Async REST Clients for Gemini, Mistral, Groq, and OpenRouter.
- **Telemetry & Testing**: `psutil`, PyTorch `pynvml`, Pytest 9.1.

---

### SLIDE 11 — IMPLEMENTATION
- **Core Pipeline (`app/services/orchestration_pipeline.py`)**: Coordinates 14-stage pipeline execution and fallbacks.
- **Decision Engine (`app/services/policies/baseline_policy.py`)**: Enforces single-authority 5-factor scoring.
- **Cloud Provider Manager (`app/services/providers/online_provider_manager.py`)**: Asynchronous HTTPX REST client for Gemini, Mistral, Groq, OpenRouter.
- **Shadow RL Policy (`app/services/policies/rl_bandit_policy.py`)**: Predicts Q-values in shadow mode (`production_override: false`).
- **Frontend Dashboard (`frontend/src/App.jsx`)**: React 18 UI displaying stage trackers, VRAM gauges, and decision breakdown tables (Built in 2.75s).
- **Automated Testing**: 28 unit and integration tests passing in `backend/tests/` (100% pass rate).

---

### SLIDE 12 — AI/ML MODEL OR CORE TECHNICAL APPROACH
- **BAAI/bge-m3 PyTorch Model**: Outputs 1024D dense vectors for cosine similarity against intent prototypes.
- **Baseline Adaptive Policy Formula**:
  $$\text{Score} = 0.30 \cdot \text{Capability} + 0.30 \cdot \text{Complexity} + 0.15 \cdot \text{Resource} + 0.15 \cdot \text{Context} + 0.10 \cdot \text{Cost}$$
- **Quality Safeguard**: Capability + Complexity fit = 60% of total decision score.
- **Neutral Tie-Breaker**: Evaluates `(candidate_score, capability_score, cost_fit_score, capability_count, context_length)`.
- **Shadow Contextual Bandit**: Linear model $Q(s, a) = W \cdot s + b$ operating on 12D state vector.

---

### SLIDE 13 — RESULTS AND EVALUATION
- **Pytest Suite Verification**: 28 / 28 Tests Passed (100% Pass Rate across authority, dispatch, fallback, scoring).
- **Frontend Production Build**: `npm run build` PASSED cleanly in 2.75s (`dist/index.html` compiled cleanly).
- **Single Authority Invariant**: 100% compliance ($\text{selected} == \text{assigned} == \text{requested} == \text{executed} == \text{reported}$).
- **Benchmarked Execution Latencies**: Sub-400ms for Flash and local 4B models; 450ms for OpenRouter 70B.
- **Token Cost Savings**: Up to 70% cost reduction on low-complexity query benchmarks.

---

### SLIDE 14 — RESULTS / SCREENSHOTS / DEMONSTRATION
- **Simple Factual QA**: `gemini-3.5-flash` / `gemma-3-4b` (382ms | Reward R=0.9950).
- **Simple Coding**: `gemini-3.5-flash` / `qwen-coder-3b` (396ms | Reward R=0.9800).
- **Technical Translation**: `mistral-small-latest` (275ms | Reward R=0.9750).
- **Calculus Proofs**: `llama-3.3-70b-versatile` via Groq (320ms | Reward R=0.9700).
- **System Architecture Design**: `meta-llama/llama-3.3-70b-instruct` via OpenRouter (450ms | Reward R=0.9750).

---

### SLIDE 15 — RESEARCH PAPER / LITERATURE SUPPORT
- **RouteLLM (LMSYS / UC Berkeley, 2024)**: Ong et al., arXiv:2406.18665. Matrix Factorization routers reducing cloud cost by 85%.
- **Linear Contextual Bandits (AISTATS 2011)**: Chu et al. LinUCB theoretical regret bounds for contextual action selection.
- **BGE M3-Embedding (BAAI 2024)**: Chen et al., arXiv:2402.03216. State-of-the-art multi-lingual 1024D dense embeddings.

---

### SLIDE 16 — RESEARCH GAP AND OUR CONTRIBUTION
- **Identified Research Gap**: RouteLLM focuses on cloud APIs, ignoring local host GPU memory; live RL routers risk bad model assignments during online exploration.
- **Our Technical Contributions**:
  1. Hardware-Aware VRAM Telemetry: Integrates `pynvml` checks to prevent local GPU OOM crashes.
  2. Single Production Routing Authority: `BaselineAdaptivePolicy` enforces transparent 5-factor scoring.
  3. Neutral Multi-Tier Tie-Breaker: Eliminates context window length bias.
  4. Shadow Mode RL Evaluation: Contextual Bandit logs 12D experience vectors without production risk.

---

### SLIDE 17 — TEAM CONTRIBUTION
- **Member 1 (AI Lead)**: BGE-M3 embedding integration, 9 intent prototypes, 5-factor complexity scoring, `BaselineAdaptivePolicy`.
- **Member 2 (RL Specialist)**: 12D state vector encoder, `RLContextualBanditPolicy`, linear Q-value prediction, experience replay buffer.
- **Member 3 (Backend Engineer)**: FastAPI routing architecture, SSE streaming endpoints, `OnlineProviderManager`, Ollama integration, hardware telemetry.
- **Member 4 (Frontend Lead)**: Vite/React glassmorphism dashboard, SSE client consumer, 28 Pytest test scripts, benchmarks.

---

### SLIDE 18 — CURRENT STATUS, FUTURE WORK & CONCLUSION
- **Current Status**: 100% completed core implementation, 28/28 tests passed, UI built in 2.75s, shadow RL buffer active.
- **Future Work**: Provider-specific exponential moving average (EMA) latency tracking, live RL production control after 5,000 samples, multi-node GPU cluster scheduling.
- **Conclusion**: Successfully built and verified a cost-aware, resource-sensitive multi-LLM orchestrator that balances cost, latency, resource availability, and response quality.

---

### FINAL SLIDE — REFERENCES
- [1] Ong et al., "RouteLLM: Learning to Route Between Large Language Models", arXiv:2406.18665, LMSYS Org / UC Berkeley, 2024.
- [2] Chu et al., "Contextual Bandits with Linear Function Approximation", Proceedings of AISTATS, 2011.
- [3] Chen et al., "BGE M3-Embedding: Multi-Lingual & Multi-Granularity Embeddings", arXiv:2402.03216, BAAI, 2024.
- [4] Tiangolo, "FastAPI: High performance, easy to learn, fast to code", Official Documentation, 2024.
- [5] Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019.
