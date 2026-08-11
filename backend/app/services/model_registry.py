import os
import json
import logging
from typing import Dict, List, Optional
from urllib.parse import unquote

from app.core.config import settings
from app.schemas.model import ModelMetadata, ModelRegistryResponse

logger = logging.getLogger("orchestrator")

class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, registry_path: Optional[str] = None):
        if self._initialized:
            return

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        raw_path = registry_path or os.path.join("datasets", "models", "model_registry.json")
        self.registry_path = (
            raw_path if os.path.isabs(raw_path)
            else os.path.join(project_root, raw_path)
        )

        self._models: Dict[str, ModelMetadata] = {}
        self._load_registry()
        self._initialized = True

    def _load_registry(self):
        if not os.path.exists(self.registry_path):
            logger.warning(f"Model registry file not found at '{self.registry_path}'")
            return

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                raw_entries = json.load(f)

            self._models = {}
            for entry in raw_entries:
                model_meta = ModelMetadata(**entry)
                # Dynamically resolve availability based on real runtime environment settings
                model_meta = self._resolve_runtime_status(model_meta)
                self._models[model_meta.model_id] = model_meta

            logger.info(f"Loaded {len(self._models)} models into ModelRegistry from '{self.registry_path}'")
        except Exception as e:
            logger.error(f"Failed to load ModelRegistry from '{self.registry_path}': {str(e)}")
            raise RuntimeError(f"Could not load ModelRegistry: {str(e)}")

    def _resolve_runtime_status(self, model: ModelMetadata) -> ModelMetadata:
        provider_lower = (model.provider or "").lower()
        if model.execution_mode == "online_api" or "api" in provider_lower:
            if "gemini" in provider_lower or "google" in provider_lower:
                has_key = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip())
            elif "mistral" in provider_lower:
                has_key = bool(settings.MISTRAL_API_KEY and settings.MISTRAL_API_KEY.strip())
            elif "groq" in provider_lower:
                has_key = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())
            elif "openrouter" in provider_lower:
                has_key = bool(settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY.strip())
            else:
                has_key = True

            model.available = has_key
            model.configuration_status = "configured" if has_key else "not_configured"
        elif model.execution_mode == "local" and model.model_id == "BAAI/bge-m3":
            model.available = True
            model.configuration_status = "configured"
        elif model.provider.lower() == "ollama" or model.execution_mode == "local":
            model.available = True
            model.configuration_status = "configured"

        return model

    def list_models(self) -> List[ModelMetadata]:
        # Re-resolve runtime status in case settings changed dynamically
        return [self._resolve_runtime_status(model) for model in self._models.values()]

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        decoded_id = unquote(model_id)
        model = self._models.get(decoded_id) or self._models.get(model_id)
        if model:
            return self._resolve_runtime_status(model)
        return None

    def is_available(self, model_id: str) -> bool:
        model = self.get_model(model_id)
        return model.available if model else False

    def register_model(self, metadata: ModelMetadata):
        resolved = self._resolve_runtime_status(metadata)
        self._models[resolved.model_id] = resolved
        logger.info(f"Registered model '{resolved.model_id}' in ModelRegistry")
