# Adaptive Multi-LLM Orchestration Using Contextual Bandits

### Complete Project Summary — What We Have Done + What We Are Going to Do

Your project is essentially an **intelligent LLM router/orchestrator**. Instead of sending every query to the same large and expensive model, the system analyzes the query, checks its complexity and the available hardware/resources, evaluates suitable models, and dynamically selects an appropriate model.

The important part is that **the current production system is controlled by the Baseline Adaptive Policy**, while the **Contextual Bandit works in shadow mode** and learns from the same decisions without directly controlling production.

---

# 1. Project Title

## **Adaptive Multi-LLM Orchestration Using Contextual Bandits**

### Subtitle

**Cost-Aware and Resource-Sensitive Dynamic Model Selection**

---

# 2. Core Idea

The basic problem is:

> **Different queries require different levels of intelligence.**

For example:

**Simple query:**

> What is the capital of France?

There is no reason to use an expensive 70B model.

**Complex query:**

> Design a fault-tolerant distributed system for dynamically routing 100,000 LLM requests while optimizing latency, cost and GPU utilization.

This requires a much more capable model.

So our system asks:

> **"Given this query, the available models, current hardware resources, cost and expected complexity, which model should handle it?"**

That is the central idea of the project.

---

# 3. Problem Statement

Current LLM applications commonly use:

* one fixed model
* manually selected models
* simple fallback mechanisms
* static routing rules

This creates several problems:

### 1. Cost

Using a powerful cloud model for every query unnecessarily increases API cost.

### 2. Latency

Large models can take longer to respond to simple requests.

### 3. Resource utilization

Local models may require significant GPU VRAM.

A model may be technically available but unsuitable when GPU memory is already heavily utilized.

### 4. Quality

Small models are inexpensive and fast, but they may perform poorly on:

* advanced mathematics
* complex reasoning
* programming
* architecture
* multi-step analysis

### 5. Lack of adaptive learning

A static router does not continuously learn from previous model executions.

---

# 4. Our Proposed Solution

We built a **Multi-LLM Orchestration System**.

The system contains:

```text
                    USER QUERY
                        │
                        ▼
                 Query Processing
                        │
                        ▼
                 BGE-M3 Embedding
                        │
                        ▼
                 Intent Detection
                        │
                        ▼
               Complexity Analysis
                        │
                        ▼
              Hardware Telemetry
                        │
                        ▼
             Available Model Discovery
                        │
                        ▼
              Candidate Filtering
                        │
                        ▼
              Baseline Adaptive Policy
                        │
                        ▼
                 Model Selection
                        │
                        ▼
                 Model Execution
                        │
                        ▼
                Response Verification
                        │
                        ▼
                 Reward Calculation
                        │
                        ▼
                 Experience Buffer
                        │
                        ▼
             Shadow Contextual Bandit
                        │
                        ▼
                 Dashboard / SSE
```

---

# 5. What We Have Already Implemented

## A. Multi-model architecture

The system supports different model providers rather than depending on a single LLM.

The architecture includes:

* Local Ollama models
* Gemini
* Mistral
* Groq
* OpenRouter

This allows the router to choose between models with different:

* capabilities
* costs
* latency
* context windows
* resource requirements

---

# 6. BGE-M3 Embedding System

We integrated:

### **BAAI/bge-m3**

The embedding service generates a:

> **1024-dimensional dense vector**

for the user query.

The purpose is to convert the natural-language query into a numerical representation that can be used for downstream analysis.

Conceptually:

```text
User Query
    ↓
BGE-M3
    ↓
1024-D Embedding
    ↓
Semantic Representation
```

---

# 7. Intent Detection

We created an intent classification mechanism using prototype-based similarity.

The system compares the query embedding against predefined intent representations.

The project currently uses **9 intent categories/prototypes**.

This allows the system to determine what type of task the user is asking for.

For example:

```text
Query
 ↓
Embedding
 ↓
Similarity Comparison
 ↓
Intent
```

Possible categories include areas such as:

* factual QA
* coding
* translation
* mathematical reasoning
* system design

The exact intent influences model selection.

---

# 8. Complexity Analysis

The system doesn't simply ask:

> "What type of query is this?"

It also asks:

> **"How difficult is this query?"**

We implemented a **5-factor complexity analysis**.

The complexity information contributes to the routing decision.

For example:

```text
Simple
   ↓
Small / fast model

Medium
   ↓
Medium-capability model

Complex
   ↓
High-capability model
```

This prevents unnecessarily sending simple requests to expensive models while ensuring difficult requests are handled by capable models.

---

# 9. Hardware Telemetry

One of the important features of our project is **resource awareness**.

The system checks host resources such as:

* CPU utilization
* RAM utilization
* GPU utilization
* free GPU VRAM

The project uses telemetry tools including:

* `psutil`
* `pynvml` / GPU telemetry mechanisms

This is important for local models.

For example:

```text
Model requires 8 GB VRAM
        ↓
Available VRAM = 10 GB
        ↓
Candidate is feasible
```

But:

```text
Model requires 8 GB VRAM
        ↓
Available VRAM = 4 GB
        ↓
Candidate rejected
```

Therefore the router is not only **query-aware**, but also **hardware-aware**.

---

# 10. Model Registry

We created a model registry:

```text
datasets/models/model_registry.json
```

This contains metadata about registered models.

The registry allows the system to understand information such as:

* model identity
* provider
* capability
* pricing
* resource requirements
* suitability information

The project presentation currently describes **8 registered local/cloud models**.

---

# 11. Candidate Model Filtering

After analyzing the query, the system identifies possible models.

Models that are clearly unsuitable can be removed before scoring.

For example:

```text
All Models
    ↓
Capability Filter
    ↓
Complexity Filter
    ↓
Context Filter
    ↓
Resource Filter
    ↓
Cost / Constraint Filter
    ↓
Candidate Models
```

This prevents the router from selecting an obviously unsuitable model.

---

# 12. Baseline Adaptive Policy

This is currently the **main production routing authority**.

The policy calculates a score for candidate models.

The scoring formulation used in the project is:

```text
Score =
0.30 × Capability
+ 0.30 × Complexity
+ 0.15 × Resource
+ 0.15 × Context
+ 0.10 × Cost
```

The important point is:

### Capability + Complexity = 60%

So the system prioritizes whether the model can actually solve the task.

Cost and resources are important, but we do not want to select a cheap model that cannot solve the query.

---

# 13. Single-Authority Architecture

One of the important design decisions we implemented is the **single-authority routing principle**.

The production path is:

```text
Query
 ↓
Baseline Adaptive Policy
 ↓
Selected Model
 ↓
Execution
```

The selected model remains consistent across the pipeline.

We also tested the invariant:

```text
selected
   =
assigned
   =
requested
   =
executed
   =
reported
```

This prevents situations where the router says one model was selected but another model actually executes the request.

---

# 14. Contextual Bandit

This is the machine-learning/reinforcement-learning component of the project.

We created a **Contextual Bandit policy**.

The bandit considers:

```text
State + Action
       ↓
     Q(s,a)
       ↓
Predicted Model Value
```

Here:

* `s` = current query/system state
* `a` = candidate model
* `Q(s,a)` = predicted value of selecting that model

The project uses a linear formulation:

```text
Q(s,a) = Wₐ · s + bₐ
```

---

# 15. 12-Dimensional State Vector

The contextual bandit receives a **12-dimensional state representation**.

The current presentation describes the state as containing:

1. Intent ID
2. Complexity Score
3. Query Length
4. Context Required
5. CPU Utilization
6. RAM Utilization
7. Free GPU VRAM
8. GPU Utilization
9. Latency Expectation
10. Cost Sensitivity
11. Model Capability Demand
12. History Feature

These features combine:

### Query information

with

### System information

and

### Model-selection information.

That is what makes the approach **contextual** rather than simply choosing a model globally.

---

# 16. Shadow RL Safety Mechanism

This is an especially important part of the project.

The Contextual Bandit is **NOT currently allowed to directly control production routing**.

Instead:

```text
                    Query
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
 Production Path             Shadow Path
          │                       │
          ▼                       ▼
 Baseline Policy           Contextual Bandit
          │                       │
          ▼                       ▼
 Actual Model             Predicted Model
          │                       │
          │                       ▼
          │                Logged Separately
          │
          ▼
       Response
```

This gives us a safe experimental environment.

The RL system can learn from production experiences without accidentally sending users to a poorly performing model.

---

# 17. Experience Buffer

The system stores routing experiences in:

```text
data/rl/experience_buffer.jsonl
```

The buffer records information such as:

* state
* selected action/model
* reward
* completion status
* routing experience

This creates a dataset for future contextual-bandit training and evaluation.

---

# 18. Reward System

After a model executes the query, the system calculates a reward.

The reward represents how successful the execution was.

It can account for factors such as:

* response quality
* latency
* cost
* successful completion

The presentation currently uses reward values in the range:

```text
R ∈ [0,1]
```

A high reward means the selected model performed well under the relevant conditions.

---

# 19. Real-Time Telemetry

We also implemented a real-time monitoring mechanism.

The backend can stream telemetry through:

### SSE — Server-Sent Events

The architecture is approximately:

```text
Backend
   ↓
SSE
   ↓
React Dashboard
   ↓
Live Routing Information
```

This allows the dashboard to display information about the orchestration process.

---

# 20. Backend

The backend is built using:

### FastAPI

Important components include services such as:

```text
orchestration_pipeline.py
baseline_policy.py
embedding_service.py
online_provider_manager.py
rl_bandit_policy.py
```

The backend handles:

* query processing
* routing
* model selection
* provider communication
* telemetry
* reward generation
* RL experience storage
* SSE streaming

---

# 21. Frontend

The frontend is built using:

* React 18
* Vite
* JavaScript
* HTML
* CSS

The project includes a technical dashboard with a modern UI.

It consumes the backend's telemetry stream and provides visibility into the orchestration process.

---

# 22. Testing

We created a test suite for the backend.

The presentation currently reports:

> **28 / 28 tests passed**

That corresponds to:

> **100% pass rate**

We also validated the single-authority routing invariant.

---

# 23. Performance Results

The current project presentation reports:

### UI build

Approximately:

> **2.75 seconds**

### Decision scoring

Approximately:

> **sub-15 ms**

### Example executions

The benchmark examples currently documented include:

| Query Type    | Example Model          | Latency | Reward |
| ------------- | ---------------------- | ------: | -----: |
| Factual QA    | Gemini / Gemma         | ~382 ms |  0.995 |
| Coding        | Gemini / Qwen Coder    | ~396 ms |  0.980 |
| Translation   | Mistral                | ~275 ms |  0.975 |
| Calculus      | Llama 70B / Groq       | ~320 ms |  0.970 |
| System Design | Llama 70B / OpenRouter | ~450 ms |  0.975 |

These should be presented as **our recorded benchmark examples**, rather than claiming they are universal performance guarantees.

---

# 24. Research Papers Supporting the Project

The project is supported by research around three major areas.

### 1. RouteLLM

Relevant because it demonstrates intelligent routing between LLMs based on capability/preferences and cost-quality trade-offs.

### 2. LinUCB / Contextual Bandits

Relevant because our project uses contextual-bandit concepts for model selection.

### 3. BGE-M3

Relevant because BGE-M3 provides the embedding representation used for semantic query analysis.

---

# 25. Research Gap

The research direction we are targeting is the combination of:

```text
Query Intelligence
        +
Multi-LLM Routing
        +
Cost Awareness
        +
Hardware Awareness
        +
Contextual Bandits
        +
Safe Shadow Evaluation
```

Our important engineering contribution is the integration of these components into one orchestration pipeline.

In particular, we emphasize:

### Hardware-aware routing

The router considers local GPU availability.

### Single-authority production routing

Only the validated baseline policy controls production.

### Shadow contextual bandit

RL can learn and evaluate alternative decisions without directly affecting production.

---

# 26. Complete Methodology

For your PPT, this is the clean methodology:

```text
1. Receive User Prompt
          ↓
2. Input Sanitization
          ↓
3. BGE-M3 Embedding
          ↓
4. Intent Identification
          ↓
5. Complexity Analysis
          ↓
6. Hardware Telemetry
          ↓
7. Model Discovery
          ↓
8. Candidate Filtering
          ↓
9. Model Suitability Scoring
          ↓
10. Model Selection
          ↓
11. Model Execution
          ↓
12. Response Verification
          ↓
13. Reward Calculation
          ↓
14. Experience Storage
          ↓
15. Shadow Contextual Bandit Prediction
          ↓
16. Real-Time SSE Telemetry
```

---

# 27. Technology Stack

### Programming

* Python
* JavaScript
* HTML
* CSS

### Backend

* FastAPI
* Uvicorn
* Pydantic
* SSE

### AI/ML

* PyTorch
* BGE-M3
* NumPy
* Scikit-learn
* Contextual Bandit

### LLM Providers

* Ollama
* Gemini
* Mistral
* Groq
* OpenRouter

### Hardware Monitoring

* psutil
* GPU telemetry / pynvml

### Frontend

* React
* Vite

### Testing

* Pytest

---

# 28. Team of Four

For the presentation, the four-member contribution structure is:

### Member 1 — AI / Routing

Responsible for:

* BGE-M3 integration
* intent prototypes
* complexity analysis
* Baseline Adaptive Policy
* model-selection logic

### Member 2 — RL / Bandit

Responsible for:

* 12-D state vector
* contextual-bandit implementation
* Q-value prediction
* reward/experience handling
* replay buffer

### Member 3 — Backend

Responsible for:

* FastAPI architecture
* provider adapters
* Ollama integration
* hardware telemetry
* SSE/backend infrastructure

### Member 4 — Frontend / Evaluation

Responsible for:

* React dashboard
* SSE client
* visualization
* testing
* benchmark/evaluation

---

# 29. What We Are Doing NOW

The project has moved beyond the basic implementation stage.

The current focus is:

### 1. Validate the complete pipeline

We need to demonstrate that:

```text
Prompt
 ↓
Analysis
 ↓
Routing
 ↓
Model
 ↓
Response
 ↓
Reward
 ↓
RL Experience
```

works correctly end-to-end.

### 2. Demonstrate different query complexities

We should test:

**Low**

> What is the capital of France?

**Medium**

> Explain how binary search works and analyze its complexity.

**High**

> Design a fault-tolerant multi-LLM router that minimizes cost and latency while maintaining response quality under changing GPU availability and API failures.

This demonstrates that the router behaves differently depending on query requirements.

---

# 30. What We Are Going to Do Next

## Phase 1 — More Extensive Evaluation

We need a larger benchmark rather than only a few examples.

Create a test set containing:

* simple queries
* medium queries
* complex queries
* coding
* mathematics
* reasoning
* translation
* system design
* factual QA

Then compare routing decisions.

---

# 31. Phase 2 — Compare Routing Strategies

This is important for the research aspect.

Compare:

### Baseline 1

Fixed large model.

### Baseline 2

Fixed small model.

### Baseline 3

Simple capability-based routing.

### Our System

Baseline Adaptive Policy.

### Experimental System

Contextual Bandit.

Then measure:

* quality
* cost
* latency
* resource usage
* failure rate

---

# 32. Phase 3 — Improve RL Training

Currently:

```text
Contextual Bandit
       ↓
Shadow Mode
```

The next stage is to collect a sufficiently large experience dataset.

The presentation currently proposes:

> **5,000 samples**

before considering production control.

The idea is:

```text
Shadow RL
   ↓
Collect Experiences
   ↓
Evaluate RL
   ↓
Compare Against Baseline
   ↓
Controlled Testing
   ↓
Production Deployment
```

---

# 33. Phase 4 — Latency Tracking

A future improvement is:

### Provider-specific EMA latency tracking

Instead of treating latency as a static value, continuously update the expected latency:

```text
Previous Latency
       +
New Observation
       ↓
EMA
       ↓
Updated Latency Estimate
```

The router can then make better decisions based on current provider performance.

---

# 34. Phase 5 — Better Cost Optimization

The system can eventually learn:

```text
Query Difficulty
       +
Required Quality
       +
Model Capability
       +
Current Cost
       ↓
Optimal Model
```

The goal is not simply:

> **Always choose the cheapest model.**

The goal is:

> **Choose the least expensive model that can reliably satisfy the query requirements.**

---

# 35. Phase 6 — Multi-GPU / Multi-Node Expansion

A longer-term improvement is extending the system from one machine to multiple machines.

Instead of:

```text
One GPU
 ↓
Local Models
```

we can have:

```text
                Router
                  ↓
       ┌──────────┼──────────┐
       ↓          ↓          ↓
    GPU Node 1 GPU Node 2 GPU Node 3
       ↓          ↓          ↓
    Models      Models      Models
```

The router could then consider:

* GPU availability
* network latency
* model availability
* workload
* cost

---

# 36. Final Expected System

The final vision is:

```text
                    USER
                     │
                     ▼
                 LLM Router
                     │
          ┌──────────┴──────────┐
          │                     │
     Query Analysis       System Telemetry
          │                     │
          └──────────┬──────────┘
                     ▼
             Adaptive Selection
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Small LLM   Medium LLM   Large LLM
          │          │          │
          └──────────┼──────────┘
                     ▼
                  Response
                     │
                     ▼
                   Reward
                     │
                     ▼
             Contextual Bandit
                     │
                     ▼
              Continuous Learning
```

---

# 37. One-Line Explanation for Review

If your faculty asks **"What exactly is your project?"**, say:

> **“Our project is an adaptive multi-LLM orchestration system that analyzes the user's query, estimates its intent and complexity, considers available computational resources and model characteristics, and dynamically selects a suitable LLM while optimizing quality, cost and latency. A contextual bandit learns from these routing experiences in shadow mode without directly affecting production decisions.”**

---

# 38. What Makes Our Project Different

The strongest points to emphasize are:

### **1. Multi-LLM**

We don't depend on one model.

### **2. Adaptive**

The model is selected according to the query.

### **3. Cost-aware**

Expensive models are not unnecessarily used.

### **4. Resource-aware**

GPU/CPU/RAM conditions are considered.

### **5. Context-aware**

Query characteristics are represented in the routing state.

### **6. Contextual Bandit**

The system can learn which model performs better for different contexts.

### **7. Safe RL**

The RL policy initially operates in **shadow mode**.

### **8. Real-time monitoring**

The orchestration process can be observed through the dashboard.

---

# 39. Current Status

Based on the project information we have documented:

| Component                | Status                   |
| ------------------------ | ------------------------ |
| Multi-LLM architecture   | ✅ Implemented            |
| BGE-M3 embeddings        | ✅ Implemented            |
| Intent detection         | ✅ Implemented            |
| Complexity analysis      | ✅ Implemented            |
| Hardware telemetry       | ✅ Implemented            |
| Model registry           | ✅ Implemented            |
| Candidate filtering      | ✅ Implemented            |
| Baseline Adaptive Policy | ✅ Implemented            |
| Contextual Bandit        | ✅ Implemented            |
| 12-D state               | ✅ Implemented            |
| Shadow RL                | ✅ Implemented            |
| Experience buffer        | ✅ Implemented            |
| SSE telemetry            | ✅ Implemented            |
| React dashboard          | ✅ Implemented            |
| Backend tests            | ✅ 28/28 reported passing |
| Benchmarking             | ✅ Initial benchmarks     |
| Large-scale evaluation   | 🔄 Next                  |
| RL production control    | 🔜 Future                |
| Multi-node orchestration | 🔜 Future                |

---

# 40. Final Conclusion

The project is **not simply an LLM chatbot**.

It is an **intelligent decision-making layer placed between the user and multiple LLMs**.

The central concept is:

> **Analyze → Understand → Measure → Filter → Score → Select → Execute → Evaluate → Learn**

And the most important safety principle is:

> **Baseline Adaptive Policy controls production; Contextual Bandit learns in shadow mode.**

The next major research step is therefore **not adding random features**. It is to build a sufficiently large benchmark, compare routing strategies quantitatively, collect more contextual-bandit experiences, and demonstrate that adaptive routing provides a measurable improvement in **cost, latency, resource utilization, and/or response quality**.