from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger
from app.api.routes import (
    health,
    input,
    embedding,
    intent,
    complexity,
    resource,
    models,
    gemini,
    model_manager,
    decision,
    response,
    verification,
    reward,
    orchestration,
    experience,
    rl,
    complex,
    metrics,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Adaptive AI Orchestration Platform with Complex Task Allocation"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(input.router, prefix=settings.API_V1_STR, tags=["Input Processing"])
app.include_router(embedding.router, prefix=settings.API_V1_STR, tags=["Semantic Embedding"])
app.include_router(intent.router, prefix=settings.API_V1_STR, tags=["Intent Analysis"])
app.include_router(complexity.router, prefix=settings.API_V1_STR, tags=["Complexity Analysis"])
app.include_router(resource.router, prefix=settings.API_V1_STR, tags=["Resource Telemetry"])
app.include_router(models.router, prefix=settings.API_V1_STR, tags=["Model Registry"])
app.include_router(gemini.router, prefix=settings.API_V1_STR, tags=["Gemini Provider"])
app.include_router(model_manager.router, prefix=settings.API_V1_STR, tags=["Model Manager"])
app.include_router(decision.router, prefix=settings.API_V1_STR, tags=["Adaptive Decision Engine"])
app.include_router(response.router, prefix=settings.API_V1_STR, tags=["Response Generator"])
app.include_router(verification.router, prefix=settings.API_V1_STR, tags=["Response Verifier"])
app.include_router(reward.router, prefix=settings.API_V1_STR, tags=["Reward Signal"])
app.include_router(orchestration.router, prefix=settings.API_V1_STR, tags=["End-to-End Orchestration"])
app.include_router(experience.router, prefix=settings.API_V1_STR, tags=["Experience Replay Buffer"])
app.include_router(rl.router, prefix=settings.API_V1_STR, tags=["RL Policy Framework"])
app.include_router(complex.router, prefix=settings.API_V1_STR, tags=["Complex Task Allocation"])
app.include_router(metrics.router, prefix=settings.API_V1_STR, tags=["Cost & Performance Metrics"])

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")

    # Process-level Singleton Startup Initialization
    from app.services.embedding_service import EmbeddingService
    from app.services.intent_prototype_service import IntentPrototypeService
    from app.services.complexity_prototype_service import ComplexityPrototypeService
    from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
    from app.services.experience_buffer import ExperienceBufferService

    emb_svc = EmbeddingService()
    emb_svc._load_model()
    logger.info("[INIT] BGE-M3 loaded")

    intent_proto = IntentPrototypeService()
    intent_proto.get_prototype_embeddings()
    logger.info("[INIT] Intent prototype cache loaded")

    cmplx_proto = ComplexityPrototypeService()
    cmplx_proto.get_prototype_embeddings()
    logger.info("[INIT] Complexity prototype cache loaded")

    rl_policy = RLContextualBanditPolicy()
    logger.info("[INIT] RL policy loaded")

    exp_buffer = ExperienceBufferService()
    logger.info("[INIT] Experience buffer loaded")
