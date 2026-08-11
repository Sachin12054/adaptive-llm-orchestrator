import re
import time
import logging
from typing import Dict, List, Any
import numpy as np

from app.core.config import settings
from app.schemas.complexity import ComplexityResponse, ComplexityFactors
from app.services.embedding_service import EmbeddingService
from app.services.complexity_prototype_service import ComplexityPrototypeService

logger = logging.getLogger("orchestrator")

class ComplexityAnalyzer:
    def __init__(
        self,
        weight_semantic: float = None,
        weight_reasoning: float = None,
        weight_task: float = None,
        weight_context: float = None,
        weight_output: float = None,
        low_threshold: float = None,
        medium_threshold: float = None,
        high_threshold: float = None
    ):
        self.w_semantic = weight_semantic if weight_semantic is not None else settings.COMPLEXITY_WEIGHT_SEMANTIC
        self.w_reasoning = weight_reasoning if weight_reasoning is not None else settings.COMPLEXITY_WEIGHT_REASONING
        self.w_task = weight_task if weight_task is not None else settings.COMPLEXITY_WEIGHT_TASK
        self.w_context = weight_context if weight_context is not None else settings.COMPLEXITY_WEIGHT_CONTEXT
        self.w_output = weight_output if weight_output is not None else settings.COMPLEXITY_WEIGHT_OUTPUT

        self.t_low = low_threshold if low_threshold is not None else settings.COMPLEXITY_LOW_THRESHOLD
        self.t_medium = medium_threshold if medium_threshold is not None else settings.COMPLEXITY_MEDIUM_THRESHOLD
        self.t_high = high_threshold if high_threshold is not None else settings.COMPLEXITY_HIGH_THRESHOLD

        self.embedding_service = EmbeddingService()
        self.prototype_service = ComplexityPrototypeService()

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _calculate_task_count(self, text: str) -> int:
        # Detect task conjunctions, bullet points, and distinct imperative verbs
        delimiters = r"[.;\n]|(?:\b(?:and then|also|furthermore|in addition|next|finally)\b)"
        parts = [p.strip() for p in re.split(delimiters, text, flags=re.IGNORECASE) if p.strip()]
        
        # Action verbs indicating distinct operations
        action_verbs = r"\b(?:write|design|compare|implement|analyze|derive|architect|benchmark|refactor|evaluate|calculate|solve|explain|summarize|translate|build)\b"
        verb_matches = len(re.findall(action_verbs, text, flags=re.IGNORECASE))
        
        count = max(len(parts), verb_matches, 1)
        return min(count, 10)

    def _estimate_reasoning_depth(self, text: str, sim_high: float, sim_very_high: float) -> int:
        # Multi-factor / analytical depth indicators
        high_reasoning_terms = r"\b(?:trade-offs?|tradeoffs?|pros and cons|architect(?:ure)?|design|consensus|concurrency|deadlock|migration|vulnerabilit(?:y|ies)|benchmark|optimization|distributed|microservices)\b"
        matches = len(re.findall(high_reasoning_terms, text, flags=re.IGNORECASE))

        avg_high_sim = (sim_high + sim_very_high) / 2.0

        if avg_high_sim > 0.65 or matches >= 3:
            return 4
        elif avg_high_sim > 0.55 or matches >= 2:
            return 3
        elif avg_high_sim > 0.45 or matches >= 1:
            return 2
        elif avg_high_sim > 0.35:
            return 1
        else:
            return 0

    def _estimate_output_complexity(self, text: str, reasoning_depth: int) -> float:
        # Detect explicit multi-artifact or high-volume output requests
        multi_artifact_terms = r"\b(?:architecture|blueprint|implementation strategy|trade-offs|full-stack|end-to-end|system design|code and explanation)\b"
        code_terms = r"\b(?:python|c\+\+|java|rust|sql|javascript|code|script|function|program|api)\b"

        if re.search(multi_artifact_terms, text, flags=re.IGNORECASE) or reasoning_depth >= 4:
            return 1.0
        elif re.search(code_terms, text, flags=re.IGNORECASE) or reasoning_depth == 3:
            return 0.75
        elif reasoning_depth == 2:
            return 0.50
        elif reasoning_depth == 1:
            return 0.30
        else:
            return 0.10

    def analyze_complexity(self, text: str) -> ComplexityResponse:
        if text is None or not isinstance(text, str) or not text.strip():
            raise ValueError("Text for complexity analysis cannot be empty or contain only whitespace.")

        t0 = time.perf_counter()

        # Step 1: Real BGE-M3 embedding for input text
        emb_res = self.embedding_service.generate_embedding(text, include_vector=True)
        user_vector = np.array(emb_res.embedding, dtype=np.float32)
        
        t1 = time.perf_counter()
        embedding_ms = round((t1 - t0) * 1000, 2)

        # Step 2: Prototype similarity calculation
        prototypes = self.prototype_service.get_prototype_embeddings()
        level_sims: Dict[str, float] = {}

        for level in ["low", "medium", "high", "very_high"]:
            items = prototypes.get(level, [])
            sims = [self._cosine_similarity(user_vector, item["vector"]) for item in items]
            sims.sort(reverse=True)
            top_k_sims = sims[:3] if len(sims) >= 3 else sims
            level_sims[level] = float(np.mean(top_k_sims)) if top_k_sims else 0.0

        sim_low = level_sims.get("low", 0.0)
        sim_medium = level_sims.get("medium", 0.0)
        sim_high = level_sims.get("high", 0.0)
        sim_very_high = level_sims.get("very_high", 0.0)

        # 1. Semantic Complexity factor [0.0 - 1.0]
        # Weighted expectation of similarity across complexity levels
        sem_score = (
            0.05 * sim_low +
            0.35 * sim_medium +
            0.70 * sim_high +
            1.00 * sim_very_high
        )
        semantic_complexity = round(float(min(1.0, max(0.0, sem_score))), 4)

        # 2. Task Complexity factor [0.0 - 1.0]
        task_count = self._calculate_task_count(text)
        task_complexity = round(float(min(1.0, max(0.0, (task_count - 1) / 4.0))), 4)

        # 3. Reasoning Complexity factor [0.0 - 1.0]
        reasoning_depth = self._estimate_reasoning_depth(text, sim_high, sim_very_high)
        reasoning_complexity = round(float(reasoning_depth / 4.0), 4)

        # 4. Context Complexity factor [0.0 - 1.0]
        words = text.split()
        word_count = len(words)
        has_code_block = "```" in text
        has_list = bool(re.search(r"^\s*[-*1-9]\.", text, flags=re.MULTILINE))
        
        words_norm = min(1.0, word_count / 250.0)
        structure_bonus = 0.20 if (has_code_block or has_list) else 0.0
        context_complexity = round(float(min(1.0, words_norm + structure_bonus)), 4)

        # 5. Output Complexity factor [0.0 - 1.0]
        output_complexity = round(float(self._estimate_output_complexity(text, reasoning_depth)), 4)

        # Transparent Weighted Complexity Score
        raw_score = (
            self.w_semantic * semantic_complexity +
            self.w_reasoning * reasoning_complexity +
            self.w_task * task_complexity +
            self.w_context * context_complexity +
            self.w_output * output_complexity
        )
        complexity_score = round(float(min(1.0, max(0.0, raw_score))), 4)

        # Map to Complexity Level
        if complexity_score < self.t_low:
            complexity_level = "low"
        elif complexity_score < self.t_medium:
            complexity_level = "medium"
        elif complexity_score < self.t_high:
            complexity_level = "high"
        else:
            complexity_level = "very_high"

        t2 = time.perf_counter()
        analysis_ms = round((t2 - t1) * 1000, 2)
        total_latency_ms = round((t2 - t0) * 1000, 2)

        return ComplexityResponse(
            text=text,
            complexity_level=complexity_level,
            complexity_score=complexity_score,
            factors=ComplexityFactors(
                semantic_complexity=semantic_complexity,
                reasoning_complexity=reasoning_complexity,
                task_complexity=task_complexity,
                context_complexity=context_complexity,
                output_complexity=output_complexity
            ),
            task_count=task_count,
            estimated_reasoning_depth=reasoning_depth,
            embedding_model=emb_res.model,
            embedding_dimension=emb_res.dimension,
            latency_ms=total_latency_ms,
            breakdown_latency_ms={
                "embedding_ms": embedding_ms,
                "analysis_ms": analysis_ms
            }
        )
