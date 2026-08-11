import time
import logging
from typing import Dict, List, Any
import numpy as np

from app.core.config import settings
from app.schemas.intent import IntentResponse, RankedIntent
from app.services.embedding_service import EmbeddingService
from app.services.intent_prototype_service import IntentPrototypeService

logger = logging.getLogger("orchestrator")

class IntentClassifier:
    def __init__(
        self,
        top_k: int = None,
        margin_threshold: float = None
    ):
        self.top_k = top_k if top_k is not None else settings.INTENT_TOP_K_PROTOTYPES
        self.margin_threshold = (
            margin_threshold if margin_threshold is not None else settings.INTENT_MARGIN_THRESHOLD
        )
        
        self.embedding_service = EmbeddingService()
        self.prototype_service = IntentPrototypeService()

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def classify_intent(self, text: str) -> IntentResponse:
        if text is None or not isinstance(text, str) or not text.strip():
            raise ValueError("Text for intent classification cannot be empty or contain only whitespace.")

        t0 = time.perf_counter()

        # Step 1: Generate BGE-M3 embedding for input text
        emb_response = self.embedding_service.generate_embedding(text, include_vector=True)
        user_vector = np.array(emb_response.embedding, dtype=np.float32)
        
        t1 = time.perf_counter()
        embedding_latency_ms = round((t1 - t0) * 1000, 2)

        # Step 2: Fetch prototype embeddings
        prototypes = self.prototype_service.get_prototype_embeddings()

        # Step 3: Calculate cosine similarities and aggregate per intent
        intent_scores: Dict[str, float] = {}

        for intent, items in prototypes.items():
            sims = [
                self._cosine_similarity(user_vector, item["vector"])
                for item in items
            ]
            sims.sort(reverse=True)
            # Take mean of top-k prototype similarities for this intent
            top_k_sims = sims[: self.top_k] if len(sims) >= self.top_k else sims
            intent_scores[intent] = float(np.mean(top_k_sims)) if top_k_sims else 0.0

        # Step 4: Rank intents by score descending
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)

        top_intent, top_score = sorted_intents[0] if sorted_intents else ("unknown", 0.0)
        second_intent, second_score = sorted_intents[1] if len(sorted_intents) > 1 else ("none", 0.0)

        top_similarity = round(top_score, 4)
        second_similarity = round(second_score, 4)
        margin = round(top_similarity - second_similarity, 4)

        # Ambiguity is flagged if separation margin between top intent and runner-up is below margin threshold
        is_ambiguous = bool(margin < self.margin_threshold)

        ranked_intents = [
            RankedIntent(intent=intent, score=round(score, 4))
            for intent, score in sorted_intents
        ]

        t2 = time.perf_counter()
        classification_latency_ms = round((t2 - t1) * 1000, 2)
        total_latency_ms = round((t2 - t0) * 1000, 2)

        return IntentResponse(
            text=text,
            intent=top_intent,
            top_similarity=top_similarity,
            second_similarity=second_similarity,
            margin=margin,
            is_ambiguous=is_ambiguous,
            ranked_intents=ranked_intents,
            embedding_model=emb_response.model,
            embedding_dimension=emb_response.dimension,
            latency_ms=total_latency_ms,
            breakdown_latency_ms={
                "embedding_ms": embedding_latency_ms,
                "classification_ms": classification_latency_ms
            }
        )
