# Adaptive AI Orchestration Platform — Phase 1

Research-oriented web application designed to route user prompts intelligently through a resource-aware adaptive decision engine.

## 🎯 Current Objective (Phase 1)

Resource-aware routing for simple and medium user queries.

The current system implements a clean, modular, single-model adaptive pipeline:

```
USER
  │
  ▼
WEB INTERFACE
  │
  ▼
INPUT PROCESSING
  │
  ▼
SEMANTIC EMBEDDING (BAAI/bge-m3)
  │
  ▼
INTENT ANALYSIS
  │
  ▼
COMPLEXITY ANALYSIS
  │
  ▼
RESOURCE ANALYSIS
  │
  ▼
ADAPTIVE DECISION ENGINE
  │
  ▼
SIMPLE OR MEDIUM PATH
  │
  ▼
MODEL MANAGER (Google Gemini API)
  │
  ▼
RESPONSE GENERATION
  │
  ▼
LIGHTWEIGHT RESPONSE VERIFICATION
  │
  ▼
FINAL RESPONSE
  │
  ▼
WEB INTERFACE
```

---

## 🔮 Future Work (Subsequent Phases)

The platform repository includes placeholders for scaling into advanced AI orchestration capabilities in future phases:

- Task decomposition (`decomposition/`)
- Workflow DAG generation (`workflow/`)
- Simulator & RL environment (`simulator/`, `rl/`)
- Multi-LLM parallel execution (`llms/`)
- Hierarchical Multi-Agent Reinforcement Learning (PPO optimization)
- Evaluation & Experiment Tracking (`evaluation/`, `experiments/`)

> [!NOTE]
> Future features are intentionally omitted from Phase 1 to guarantee a clean, un-bloated architecture.

---

## 🛠️ Technology Stack (Phase 1)

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic
- **AI / ML**: `sentence-transformers` (`BAAI/bge-m3`), `scikit-learn`, `psutil`
- **LLM Provider**: Google Gemini API
- **Frontend**: React, Vite, Tailwind CSS
- **Database**: PostgreSQL
- **Testing**: `pytest`

---

## 🚀 Quickstart

### 1. Environment Setup

Copy `.env.example` to `.env` and fill in your Gemini API key:

```bash
cp .env.example .env
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Testing

Run automated tests (using mocks for LLM APIs):

```bash
cd backend
pytest
```