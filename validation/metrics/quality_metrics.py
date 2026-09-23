import logging
from typing import Dict, Any, Tuple

from app.services.complex.task_response_validator import TaskResponseValidator

logger = logging.getLogger("validation")

class QualityMetricsEvaluator:
    """
    Quality Evaluation Engine based on 0-5 Structured Rubric.
    Weights: Relevance (0.30), Correctness (0.25), Completeness (0.25), Domain Consistency (0.20).
    """

    @staticmethod
    def evaluate_quality(
        prompt: str,
        category: str,
        generated_text: str,
        success: bool = True
    ) -> Dict[str, Any]:
        if not success or not generated_text or not generated_text.strip():
            return {
                "score_0_to_5": 0.0,
                "normalized_quality": 0.0,
                "rubric_level": "0 - Unusable / Complete Failure",
                "relevance": 0.0,
                "correctness": 0.0,
                "completeness": 0.0,
                "domain_consistency": 0.0,
                "reason": "Execution failed or empty response returned."
            }

        # Validate domain consistency using TaskResponseValidator
        is_valid, rel_score, reason = TaskResponseValidator.validate_response_relevance(
            task_id="VAL-01",
            original_user_prompt=prompt,
            task_objective=f"Answer request for category {category}",
            category=category,
            response_text=generated_text
        )

        domain_consistency_score = 1.0 if is_valid else 0.10
        length = len(generated_text.strip())

        # Completeness based on output length & structure
        completeness_score = min(1.0, length / 400.0) if category != "Resource-Sensitive" else 1.0
        relevance_score = max(0.2, rel_score)
        correctness_score = 0.85 if is_valid else 0.30

        # Weighted quality score (0.0 to 1.0)
        norm_quality = (
            relevance_score * 0.30 +
            correctness_score * 0.25 +
            completeness_score * 0.25 +
            domain_consistency_score * 0.20
        )

        score_0_to_5 = round(norm_quality * 5.0, 2)
        
        # Map to 0-5 Rubric Level
        if score_0_to_5 < 1.0:
            level = "0 - Unusable"
        elif score_0_to_5 < 2.0:
            level = "1 - Poor"
        elif score_0_to_5 < 3.0:
            level = "2 - Partially useful"
        elif score_0_to_5 < 4.0:
            level = "3 - Acceptable"
        elif score_0_to_5 < 4.7:
            level = "4 - Good"
        else:
            level = "5 - Excellent"

        return {
            "score_0_to_5": score_0_to_5,
            "normalized_quality": round(norm_quality, 4),
            "rubric_level": level,
            "relevance": round(relevance_score, 2),
            "correctness": round(correctness_score, 2),
            "completeness": round(completeness_score, 2),
            "domain_consistency": round(domain_consistency_score, 2),
            "reason": reason
        }
