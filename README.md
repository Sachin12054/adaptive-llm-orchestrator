<div align="center">

# Adaptive LLM Orchestrator

### Resource-aware, multi-provider language-model routing with verification and reinforcement-learning evaluation

FastAPI backend | React + Vite dashboard | Ollama local inference | Optional cloud providers | Contextual-bandit research workflow

</div>

## Overview

Adaptive LLM Orchestrator is a research-oriented platform for selecting and coordinating language models according to prompt intent, estimated complexity, available resources, execution mode, quality signals, latency, and cost.

Instead of sending every request to the same model, the platform analyzes each prompt and chooses an appropriate execution path:

- **Simple and medium requests** use a single selected model.
- **Complex requests** can be decomposed into typed subtasks and dependency levels.
- **Local mode** routes to models served by Ollama.
- **Online mode** can use configured Gemini, Mistral, Groq, or OpenRouter providers.
- **Verification and reward stages** measure the result and record an experience for offline policy evaluation.
- **The RL policy currently operates as a shadow policy by default**, so experimental recommendations do not silently replace the production routing policy.

The repository contains the application, datasets, evaluation runners, scientific validation reports, and presentation material for the project.

## Key Capabilities

| Capability | Description |
| --- | --- |
| Prompt processing | Normalizes input and enforces a configurable maximum prompt length. |
| Semantic embeddings | Uses `BAAI/bge-m3` through Sentence Transformers to create a 1024-dimensional representation. |
| Intent analysis | Classifies prompts against prototype classes such as coding, reasoning, mathematics, factual, summarization, explanation, creative, translation, and conversational. |
| Complexity analysis | Combines semantic, reasoning, task, context, and output complexity into a routing score. |
| Adaptive routing | Selects a model using the baseline policy and resource-aware execution information. |
| Multi-provider execution | Supports local Ollama plus optional Gemini, Mistral, Groq, and OpenRouter adapters. |
| Complex-task planning | Decomposes complex prompts, builds a dependency DAG, computes execution levels, and allocates models by subtask category. |
| Streaming | Exposes Server-Sent Events for stage-by-stage orchestration progress. |
| Verification and rewards | Verifies generated responses and calculates deterministic structural, completeness, relevance, verification, and execution rewards. |
| RL experience collection | Stores transitions with a 12-dimensional state vector for contextual-bandit analysis. |
| Observability | Reports model readiness, resource telemetry, latency, token usage, and cost metrics. |

## Architecture

```mermaid
flowchart TD
		A[User prompt] --> B[Input processor]
		B --> C[BGE-M3 embedding]
		C --> D[Intent classifier]
		C --> E[Complexity analyzer]
		B --> F[Resource telemetry]
		D --> G[Adaptive decision engine]
		E --> G
		F --> G
		G --> H{Execution mode}
		H -->|Local| I[Ollama provider]
		H -->|Online| J[Provider manager]
		J --> J1[Gemini]
		J --> J2[Mistral]
		J --> J3[Groq]
		J --> J4[OpenRouter]
		I --> K[Response verification]
		J1 --> K
		J2 --> K
		J3 --> K
		J4 --> K
		G --> L{Complex request?}
		L -->|Yes| M[Decomposer and dependency graph]
		M --> N[Task allocation / scheduling]
		N --> K
		K --> O[Reward signal]
		O --> P[Experience replay buffer]
		P --> Q[Offline RL shadow evaluation]
```

### Routing and task allocation

The default baseline routes according to prompt signals and available models. The complex-task allocator uses these category defaults:

| Subtask category | Default local model |
| --- | --- |
| Coding | `qwen-coder-3b` |
| Reasoning, mathematics, analysis | `deepseek-r1-7b` |
| General, explanation, factual, conversational, summarization | `gemma-3-4b` |
| Unknown or ambiguous | `gemma-3-4b` |

Complex plans are represented as a dependency graph with cycle detection and parallelizable execution levels. Depending on the selected pipeline and execution mode, a complex request may return a plan or continue through scheduling and synthesis.

### Reinforcement learning boundary

The contextual bandit uses a three-action local model space:

| Action | Model |
| --- | --- |
| `0` | `gemma-3-4b` |
| `1` | `qwen-coder-3b` |
| `2` | `deepseek-r1-7b` |

Production routing remains controlled by the baseline policy unless the deployment configuration explicitly changes it. Shadow evaluation and experience collection are intended for offline analysis and controlled research experiments.

## Repository Structure

```text
adaptive-llm-orchestrator/
├── backend/
│   ├── app/
│   │   ├── api/routes/          FastAPI endpoints
│   │   ├── core/                Settings and logging
│   │   ├── schemas/             Request and response models
│   │   └── services/            Routing, providers, RL, verification, and orchestration
│   ├── tests/                   Backend unit and integration tests
│   └── *.py                     Benchmarks, runners, and research utilities
├── frontend/
│   ├── src/components/          Dashboard components
│   ├── src/pages/               Requests, history, analytics, models, settings, and RL views
│   └── src/services/api.js      Backend API client
├── datasets/                    Intent and complexity prototypes and evaluation data
├── data/                        Embedding caches, logs, evaluation output, and RL buffers
├── database/                    Database-related assets
├── decomposition/                Decomposition resources
├── docs/                        Architecture and project documentation
├── evaluation/                  Benchmark runners
├── experiments/                 Audits and reproducibility scripts
├── rl/                          RL research assets
├── simulator/                   Simulation utilities
├── validation/                  Validation runners and reports
├── docker-compose.yml           PostgreSQL service definition
├── requirements.txt             Root Python dependencies
└── presentation.md              Project presentation source
```

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm
- Docker Desktop, if PostgreSQL is required
- Ollama, if local inference is required
- Enough disk space and memory for `BAAI/bge-m3` and the selected local models
- Cloud provider API keys only for online execution

## Installation

### 1. Clone and create a Python environment

PowerShell:

```powershell
git clone <repository-url>
cd adaptive-llm-orchestrator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

Linux or macOS:

```bash
git clone <repository-url>
cd adaptive-llm-orchestrator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

The root `requirements.txt` contains the broader research and experiment stack. Use it when running scripts that require packages beyond the backend service:

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the repository root from the checked-in template:

```powershell
Copy-Item .env.example .env
```

Fill in only the provider keys needed for your selected execution mode. Never commit `.env` or real API keys.

### 3. Start PostgreSQL

```bash
docker compose up -d postgres
```

The default connection is:

```text
postgresql://postgres:password@localhost:5432/adaptive_orchestrator
```

The compose file uses a persistent Docker volume named `postgres_data`.

### 4. Install and prepare Ollama models

Install Ollama from [ollama.com](https://ollama.com), start the Ollama service, and pull the models used by the local routing path. Model tags depend on the local Ollama registry and your available hardware, so verify the exact tags with `ollama list`.

```bash
ollama serve
ollama pull gemma-3-4b
ollama pull qwen-coder-3b
ollama pull deepseek-r1-7b
```

The backend expects Ollama at `http://localhost:11434` by default. Change `OLLAMA_BASE_URL` when it runs elsewhere.

## Running the Application

### Backend

Run from the repository root:

```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`.

- OpenAPI UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health endpoint: `http://127.0.0.1:8000/api/health`

The first startup loads the embedding model and prototype caches, so initialization can take longer than later requests.

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the dashboard at the URL shown in the terminal, normally `http://localhost:5173`. The frontend uses the `/api` path; configure the Vite proxy or serve it through the same origin as the backend when deploying.

Production frontend commands:

```bash
npm run lint
npm run build
npm run preview
```

## API Usage

### Check readiness

```bash
curl http://127.0.0.1:8000/api/orchestrate/status
```

### Run an orchestration request

```bash
curl -X POST http://127.0.0.1:8000/api/orchestrate \
	-H "Content-Type: application/json" \
	-d '{"prompt":"Explain binary search and provide Python code.","execution_mode":"local"}'
```

The response includes the selected decision, generated response, verification result, reward information, provider metadata, and execution metrics when available.

### Stream orchestration events

```bash
curl -N -X POST http://127.0.0.1:8000/api/orchestrate/stream \
	-H "Content-Type: application/json" \
	-d '{"prompt":"Compare BFS and DFS for a graph problem.","execution_mode":"local"}'
```

### Inspect other service surfaces

The API is grouped under `/api` and includes routes for:

- `/health` - service health
- `/input` - input processing
- `/embedding` - semantic embeddings
- `/intent` - intent classification
- `/complexity` - complexity analysis
- `/resource` - CPU, memory, GPU, and Ollama telemetry
- `/models` and `/model-manager` - model registry and readiness
- `/decision` - adaptive routing decisions
- `/response` - response generation
- `/verification` - response verification
- `/reward` - reward calculation
- `/orchestrate` - end-to-end execution and SSE streaming
- `/experience` - replay-buffer status
- `/rl` - RL policy status and evaluation
- `/complex` - complex task planning and allocation
- `/metrics` - cost and performance metrics

Use the generated OpenAPI documentation at `/docs` for exact request and response schemas.

## Configuration Reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_HOST` | `127.0.0.1` | Backend bind address. |
| `API_PORT` | `8000` | Backend port. |
| `DATABASE_URL` | Local PostgreSQL URL | Database connection string. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence Transformer model. |
| `EMBEDDING_DEVICE` | `auto` | Embedding device selection. |
| `PRODUCTION_POLICY` | `rl` | Production policy selection setting. Validate this against the deployment policy before changing it. |
| `FALLBACK_POLICY` | `baseline` | Fallback routing policy. |
| `RL_DATA_COLLECTION_MODE` | `false` | Enables controlled RL data collection. |
| `EXPLORATION_EPSILON` | `0.20` | Exploration probability for controlled collection. |
| `EXPLORATION_SEED` | `42` | Reproducible exploration seed. |
| `MAX_PROMPT_LENGTH` | `10000` | Maximum accepted prompt length. |
| `GEMINI_API_KEY` | Empty | Enables Gemini online execution. |
| `MISTRAL_API_KEY` | Empty | Enables Mistral online execution. |
| `GROQ_API_KEY` | Empty | Enables Groq online execution. |
| `OPENROUTER_API_KEY` | Empty | Enables OpenRouter online execution. |

Provider model names and prototype thresholds can also be overridden through the variables defined in `backend/app/core/config.py` and `.env.example`.

## Testing and Evaluation

Run backend tests from the `backend` directory:

```bash
pytest -q
```

Run a focused test file:

```bash
pytest -q tests/test_orchestration_pipeline.py
```

The repository also contains experiment and validation runners for provider integration, routing, cost accounting, RL experience collection, complex-task allocation, and local-only scientific evaluation. Examples include:

```bash
python experiments/run_all_tests.py
python evaluation/run_benchmark_100.py
python validation/runners/run_local_only_pipeline.py
```

Several runners make real model calls, create files under `data/`, or require Ollama and provider credentials. Read the script arguments and environment requirements before running them against production services.

## Cost and Reliability Notes

- Local Ollama execution is accounted for as zero API cost by the project cost calculator, but it still consumes local CPU, memory, GPU, and power.
- Online providers are subject to network latency, quotas, rate limits, authentication errors, and provider availability.
- Provider failover is implemented for configured online providers, but a fallback can only work when another provider is configured and available.
- Benchmark results are environment-dependent. Record hardware, model tags, provider configuration, and execution mode with each experiment.
- The default CORS configuration allows all origins for development. Restrict it before production deployment.
- The default PostgreSQL credentials in `docker-compose.yml` are for local development only.

## Security and Data Handling

- Keep `.env` outside version control and rotate any credential that may have been exposed.
- Do not place API keys in source code, notebooks, benchmark output, screenshots, or committed logs.
- Review the contents of `data/`, `logs/`, and generated evaluation artifacts before sharing the repository; prompts and model responses may be stored there.
- Replace development CORS and database settings before exposing the service beyond localhost.
- Add authentication, request limits, and provider-specific secret management before a public deployment.

## Documentation

- [System architecture](docs/architecture.md)
- [Evaluation datasets](datasets/evaluation/README.md)
- [Project presentation](presentation.md)
- Interactive API documentation at `/docs` after starting the backend

## Project Status

This is an academic and research prototype. The core end-to-end pipeline, local provider path, provider adapters, verification, reward calculation, experience buffering, complex-task planning, dashboard, and validation assets are present. Operational hardening, authentication, production secret management, database migrations, and deployment automation should be added before using it as a public service.

## License

No license file is currently included. Add a license before distributing the project outside its intended academic or internal context.
