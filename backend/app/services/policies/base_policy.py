from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from app.schemas.model import ModelMetadata
from app.schemas.decision import CandidateScoreBreakdown

class BaseDecisionPolicy(ABC):
    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Unique identifier name for this decision policy."""
        pass

    @abstractmethod
    def evaluate_candidates(
        self,
        prompt: str,
        intent_info: Dict[str, Any],
        complexity_info: Dict[str, Any],
        resource_info: Dict[str, Any],
        candidate_models: List[ModelMetadata]
    ) -> Tuple[Optional[str], float, List[CandidateScoreBreakdown], List[str]]:
        """
        Evaluates candidate models and selects the optimal model ID.
        Returns: (selected_model_id, winning_score, candidate_breakdowns, reasoning_bullets)
        """
        pass
