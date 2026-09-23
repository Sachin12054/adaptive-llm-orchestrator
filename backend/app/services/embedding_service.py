import time
import logging
from typing import Optional, List
from app.core.config import settings
from app.schemas.embedding import EmbeddingResponse

logger = logging.getLogger("orchestrator")

class EmbeddingService:
    _instance: Optional["EmbeddingService"] = None
    _shared_model = None
    _shared_dimension = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: Optional[str] = None,
        device_setting: Optional[str] = None
    ):
        if getattr(self, "_initialized", False):
            return
            
        self.model_name = model_name if model_name is not None else settings.EMBEDDING_MODEL
        self.device_setting = (
            device_setting if device_setting is not None else settings.EMBEDDING_DEVICE
        )
        self.device = self._resolve_device(self.device_setting)
        self._initialized = True

    def _resolve_device(self, setting: str) -> str:
        setting = setting.lower()
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if setting == "cuda":
            if cuda_available:
                return "cuda"
            logger.warning("CUDA requested for embedding service, but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        elif setting == "cpu":
            return "cpu"
        else:  # "auto"
            return "cuda" if cuda_available else "cpu"

    def _load_model(self):
        if EmbeddingService._shared_model is not None:
            return

        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        try:
            from sentence_transformers import SentenceTransformer
            try:
                model_inst = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                model_inst = SentenceTransformer(self.model_name, device=self.device)
            dim = model_inst.get_embedding_dimension()

            EmbeddingService._shared_model = model_inst
            EmbeddingService._shared_dimension = dim
            logger.info(f"Embedding model loaded successfully. Dimension: {dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {str(e)}")
            raise RuntimeError(f"Could not load embedding model '{self.model_name}': {str(e)}")

    @property
    def model(self):
        self._load_model()
        return EmbeddingService._shared_model

    @property
    def dimension(self):
        self._load_model()
        return EmbeddingService._shared_dimension

    def generate_embedding(self, text: str, include_vector: bool = False) -> EmbeddingResponse:
        if text is None or not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding text cannot be empty or contain only whitespace.")

        self._load_model()

        start_time = time.perf_counter()
        raw_embedding = EmbeddingService._shared_model.encode(text, convert_to_numpy=True)
        end_time = time.perf_counter()

        latency_ms = round((end_time - start_time) * 1000, 2)
        actual_dim = int(raw_embedding.shape[0]) if hasattr(raw_embedding, "shape") else len(raw_embedding)

        vector_output: Optional[List[float]] = None
        if include_vector:
            vector_output = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else list(raw_embedding)

        return EmbeddingResponse(
            model=self.model_name,
            dimension=actual_dim,
            latency_ms=latency_ms,
            device=self.device,
            embedding=vector_output
        )

    def get_embedding_dimension(self) -> int:
        self._load_model()
        return EmbeddingService._shared_dimension
