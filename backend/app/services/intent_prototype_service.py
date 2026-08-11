import os
import json
import hashlib
import logging
from typing import Dict, List, Any
import numpy as np

from app.core.config import settings
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger("orchestrator")

class IntentPrototypeService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(IntentPrototypeService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        dataset_path: str = None,
        cache_path: str = None
    ):
        if self._initialized:
            return

        # Resolve paths relative to project root if relative
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        
        raw_dataset_path = dataset_path or settings.INTENT_PROTOTYPES_DATASET
        raw_cache_path = cache_path or settings.INTENT_CACHE_PATH

        self.dataset_path = (
            raw_dataset_path if os.path.isabs(raw_dataset_path)
            else os.path.join(project_root, raw_dataset_path)
        )
        self.cache_path = (
            raw_cache_path if os.path.isabs(raw_cache_path)
            else os.path.join(project_root, raw_cache_path)
        )

        self.embedding_service = EmbeddingService()
        self.prototypes: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized = True

    def _compute_dataset_hash(self, content_str: str) -> str:
        return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    def _load_raw_dataset(self) -> Dict[str, List[str]]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Intent prototype dataset not found at '{self.dataset_path}'")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def get_prototype_embeddings(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.prototypes:
            return self.prototypes

        raw_dataset = self._load_raw_dataset()
        dataset_str = json.dumps(raw_dataset, sort_keys=True)
        dataset_hash = self._compute_dataset_hash(dataset_str)

        # Check if persistent cache is valid
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                if (
                    cache_data.get("model") == settings.EMBEDDING_MODEL and
                    cache_data.get("dataset_hash") == dataset_hash and
                    "prototypes" in cache_data
                ):
                    logger.info(f"Loaded prototype embeddings from cache '{self.cache_path}'")
                    # Convert raw lists back to numpy arrays
                    self.prototypes = {}
                    for intent, items in cache_data["prototypes"].items():
                        self.prototypes[intent] = [
                            {
                                "text": item["text"],
                                "vector": np.array(item["vector"], dtype=np.float32)
                            }
                            for item in items
                        ]
                    return self.prototypes
                else:
                    logger.info("Prototype cache exists but is stale/invalid. Rebuilding cache...")
            except Exception as e:
                logger.warning(f"Failed to read cache file '{self.cache_path}': {str(e)}. Rebuilding...")

        # Rebuild prototype embeddings using real BGE-M3 model
        logger.info(f"Generating prototype embeddings using model '{settings.EMBEDDING_MODEL}'...")
        self.prototypes = {}
        cache_payload = {
            "model": settings.EMBEDDING_MODEL,
            "dataset_hash": dataset_hash,
            "prototypes": {}
        }

        for intent, texts in raw_dataset.items():
            self.prototypes[intent] = []
            cache_payload["prototypes"][intent] = []

            for text in texts:
                emb_res = self.embedding_service.generate_embedding(text, include_vector=True)
                vec_np = np.array(emb_res.embedding, dtype=np.float32)
                
                self.prototypes[intent].append({
                    "text": text,
                    "vector": vec_np
                })
                cache_payload["prototypes"][intent].append({
                    "text": text,
                    "vector": emb_res.embedding
                })

        # Save to disk cache
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f)
            logger.info(f"Saved prototype embeddings cache to '{self.cache_path}'")
        except Exception as e:
            logger.warning(f"Could not save prototype embedding cache to '{self.cache_path}': {str(e)}")

        return self.prototypes
