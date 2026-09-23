import os
from pydantic_settings import BaseSettings

# Load .env file explicitly from project root if it exists
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Fallback manual line parsing if python-dotenv is not installed
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

class Settings(BaseSettings):
    PROJECT_NAME: str = "Adaptive AI Orchestration Platform"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"

    PRODUCTION_POLICY: str = os.getenv("PRODUCTION_POLICY", "rl")
    FALLBACK_POLICY: str = os.getenv("FALLBACK_POLICY", "baseline")

    # Controlled RL Exploration & Data Collection Mode Settings
    RL_DATA_COLLECTION_MODE: bool = os.getenv("RL_DATA_COLLECTION_MODE", "false").lower() == "true"
    EXPLORATION_EPSILON: float = float(os.getenv("EXPLORATION_EPSILON", "0.20"))
    EXPLORATION_SEED: int = int(os.getenv("EXPLORATION_SEED", "42"))

    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    MISTRAL_ENABLED: bool = os.getenv("MISTRAL_ENABLED", "true").lower() == "true"

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/adaptive_orchestrator"
    )
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "auto")

    # Intent Classification Configuration
    INTENT_TOP_K_PROTOTYPES: int = int(os.getenv("INTENT_TOP_K_PROTOTYPES", "3"))
    INTENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.50"))
    INTENT_MARGIN_THRESHOLD: float = float(os.getenv("INTENT_MARGIN_THRESHOLD", "0.025"))
    INTENT_PROTOTYPES_DATASET: str = os.getenv("INTENT_PROTOTYPES_DATASET", "datasets/intent/intent_prototypes.json")
    INTENT_CACHE_PATH: str = os.getenv("INTENT_CACHE_PATH", "data/embeddings/intent_prototypes_cache.json")

    # Complexity Analyzer Configuration
    COMPLEXITY_WEIGHT_SEMANTIC: float = float(os.getenv("COMPLEXITY_WEIGHT_SEMANTIC", "0.30"))
    COMPLEXITY_WEIGHT_REASONING: float = float(os.getenv("COMPLEXITY_WEIGHT_REASONING", "0.25"))
    COMPLEXITY_WEIGHT_TASK: float = float(os.getenv("COMPLEXITY_WEIGHT_TASK", "0.20"))
    COMPLEXITY_WEIGHT_CONTEXT: float = float(os.getenv("COMPLEXITY_WEIGHT_CONTEXT", "0.15"))
    COMPLEXITY_WEIGHT_OUTPUT: float = float(os.getenv("COMPLEXITY_WEIGHT_OUTPUT", "0.10"))

    COMPLEXITY_LOW_THRESHOLD: float = float(os.getenv("COMPLEXITY_LOW_THRESHOLD", "0.3279"))
    COMPLEXITY_MEDIUM_THRESHOLD: float = float(os.getenv("COMPLEXITY_MEDIUM_THRESHOLD", "0.4753"))
    COMPLEXITY_HIGH_THRESHOLD: float = float(os.getenv("COMPLEXITY_HIGH_THRESHOLD", "0.5564"))

    COMPLEXITY_PROTOTYPES_DATASET: str = os.getenv("COMPLEXITY_PROTOTYPES_DATASET", "datasets/complexity/complexity_prototypes.json")
    COMPLEXITY_CACHE_PATH: str = os.getenv("COMPLEXITY_CACHE_PATH", "data/embeddings/complexity_prototypes_cache.json")

    # Prompt validation limits
    MAX_PROMPT_LENGTH: int = int(os.getenv("MAX_PROMPT_LENGTH", "10000"))

    # Default Fast & Medium Gemini Model identifiers
    DEFAULT_FAST_MODEL: str = "gemini-3.6-flash"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
