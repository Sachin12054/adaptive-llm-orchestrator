import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from app.services.policies.base_policy import BaseDecisionPolicy
from app.schemas.model import ModelMetadata
from app.schemas.decision import CandidateScoreBreakdown
from app.schemas.rl import ShadowDecision
from app.services.experience_buffer import ACTION_MAP, INTENT_MAP

logger = logging.getLogger("orchestrator")

class RLContextualBanditPolicy(BaseDecisionPolicy):
    _instance: Optional["RLContextualBanditPolicy"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RLContextualBanditPolicy, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self.state_dim = 12
        self.action_map = ACTION_MAP
        self.reverse_action_map = {
            0: "gemma-3-4b",
            1: "qwen-coder-3b",
            2: "deepseek-r1-7b"
        }
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        raw_path = model_path or os.path.join("data", "rl", "models", "rl_contextual_bandit_policy.json")
        self.model_path = raw_path if os.path.isabs(raw_path) else os.path.join(project_root, raw_path)
        
        self.weights: Optional[np.ndarray] = None  # Shape (K, 12)
        self.bias: Optional[np.ndarray] = None     # Shape (K,)
        self.policy_version: Optional[str] = None
        self.training_samples_count: int = 0
        self.mean_squared_error: Optional[float] = None
        self.last_trained_timestamp: Optional[float] = None
        
        self.load_policy()
        self._initialized = True

    @property
    def policy_name(self) -> str:
        return "rl_contextual_bandit_policy"

    def load_policy(self) -> bool:
        if not os.path.exists(self.model_path):
            logger.info(f"RL Policy file not found at '{self.model_path}'. Running in uninitialized shadow status.")
            return False

        try:
            with open(self.model_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("state_dim") != 12:
                logger.error(f"RL Policy state_dim mismatch ({data.get('state_dim')} != 12).")
                return False

            weights_list = data.get("weights")
            bias_list = data.get("bias")
            if weights_list is None or bias_list is None:
                logger.error("RL Policy file missing weights or bias arrays.")
                return False

            self.weights = np.array(weights_list, dtype=np.float32)
            self.bias = np.array(bias_list, dtype=np.float32)
            self.policy_version = data.get("version", "1.0.0")
            self.training_samples_count = data.get("training_samples_count", 0)
            self.mean_squared_error = data.get("mean_squared_error")
            self.last_trained_timestamp = data.get("timestamp")

            logger.info(f"Loaded RL Contextual Bandit Policy v{self.policy_version} from '{self.model_path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to load RL Contextual Bandit Policy from '{self.model_path}': {str(e)}")
            return False

    def encode_state_vector(
        self,
        intent_info: Dict[str, Any],
        complexity_info: Dict[str, Any],
        resource_info: Dict[str, Any],
        baseline_score: float = 0.50
    ) -> List[float]:
        intent_str = intent_info.get("intent", "general_qa").lower()
        intent_code = INTENT_MAP.get(intent_str, 0) / 9.0
        is_ambiguous = 1.0 if intent_info.get("is_ambiguous", False) else 0.0

        complexity_score = float(complexity_info.get("complexity_score", 0.50))
        sem_comp = float(complexity_info.get("semantic_complexity", complexity_score))
        reas_comp = float(complexity_info.get("reasoning_complexity", complexity_score))
        task_comp = float(complexity_info.get("task_complexity", complexity_score))
        ctx_comp = float(complexity_info.get("context_complexity", complexity_score))
        out_comp = float(complexity_info.get("output_complexity", complexity_score))

        cpu_util = float(resource_info.get("cpu_utilization_percent", 20.0)) / 100.0
        mem_util = float(resource_info.get("memory_utilization_percent", 50.0)) / 100.0
        gpu_avail = 1.0 if resource_info.get("gpu_available", False) else 0.0

        return [
            round(intent_code, 4),
            round(is_ambiguous, 4),
            round(complexity_score, 4),
            round(sem_comp, 4),
            round(reas_comp, 4),
            round(task_comp, 4),
            round(ctx_comp, 4),
            round(out_comp, 4),
            round(cpu_util, 4),
            round(mem_util, 4),
            round(gpu_avail, 4),
            round(float(baseline_score), 4)
        ]

    def predict_shadow_decision(
        self,
        state_vector: List[float],
        baseline_selected_model: Optional[str],
        candidate_models: List[ModelMetadata]
    ) -> ShadowDecision:
        if self.weights is None or self.bias is None:
            return ShadowDecision(
                policy_name=self.policy_name,
                proposed_model=None,
                predicted_reward=0.0,
                action_index=-1,
                agrees_with_baseline=False,
                action_masked=False,
                policy_available=False
            )

        if len(state_vector) != self.state_dim:
            logger.warning(f"State vector dimension mismatch: expected {self.state_dim}, got {len(state_vector)}")
            return ShadowDecision(
                policy_name=self.policy_name,
                proposed_model=None,
                predicted_reward=0.0,
                action_index=-1,
                agrees_with_baseline=False,
                action_masked=False,
                policy_available=False
            )

        s_vec = np.array(state_vector, dtype=np.float32)
        k_actions = self.weights.shape[0]

        # Compute Q_hat(s, a) = W_a^T s + b_a for each action index
        raw_scores = np.dot(self.weights, s_vec) + self.bias
        masked_scores = raw_scores.copy()

        action_masked = False
        # Action Masking based on candidate model availability
        for a_idx in range(k_actions):
            model_id = self.reverse_action_map.get(a_idx)
            if not model_id:
                masked_scores[a_idx] = -float("inf")
                action_masked = True
                continue

            # Find model metadata in candidates
            cand = next((m for m in candidate_models if m.model_id == model_id), None)
            if not cand or not cand.available or cand.configuration_status != "configured":
                masked_scores[a_idx] = -float("inf")
                action_masked = True

        # Check if all actions are masked
        if np.all(masked_scores == -float("inf")):
            return ShadowDecision(
                policy_name=self.policy_name,
                proposed_model=None,
                predicted_reward=0.0,
                action_index=-1,
                agrees_with_baseline=False,
                action_masked=True,
                policy_available=True
            )

        best_action_idx = int(np.argmax(masked_scores))
        best_score = float(masked_scores[best_action_idx])
        proposed_model = self.reverse_action_map.get(best_action_idx)

        agrees = (proposed_model == baseline_selected_model) if baseline_selected_model else False

        return ShadowDecision(
            policy_name=self.policy_name,
            proposed_model=proposed_model,
            predicted_reward=round(best_score, 4),
            action_index=best_action_idx,
            agrees_with_baseline=agrees,
            action_masked=action_masked,
            policy_available=True
        )

    def evaluate_candidates(
        self,
        prompt: str,
        intent_info: Dict[str, Any],
        complexity_info: Dict[str, Any],
        resource_info: Dict[str, Any],
        candidate_models: List[ModelMetadata]
    ) -> Tuple[Optional[str], float, List[CandidateScoreBreakdown], List[str]]:
        """
        Concrete implementation of BaseDecisionPolicy.evaluate_candidates.
        Note: Production routing remains controlled by BaselineAdaptivePolicy.
        This evaluation interface supports offline policy evaluation and shadow inference.
        """
        state_vec = self.encode_state_vector(intent_info, complexity_info, resource_info)
        shadow_dec = self.predict_shadow_decision(state_vec, None, candidate_models)

        breakdowns: List[CandidateScoreBreakdown] = []

        if not shadow_dec.policy_available or not shadow_dec.proposed_model:
            reasoning = [
                f"RL Policy '{self.policy_name}' evaluated prompt in shadow mode.",
                "Policy weights are currently uninitialized or unconfigured.",
                "No shadow RL candidate model proposed."
            ]
            for model in candidate_models:
                breakdowns.append(CandidateScoreBreakdown(
                    model_id=model.model_id,
                    eligible=False,
                    ineligible_reason="RL Policy is running in uninitialized shadow mode.",
                    candidate_score=0.0
                ))
            return None, 0.0, breakdowns, reasoning

        proposed_id = shadow_dec.proposed_model
        predicted_score = shadow_dec.predicted_reward

        for model in candidate_models:
            is_proposed = (model.model_id == proposed_id)
            is_eligible = (model.model_type == "llm" and model.available and model.configuration_status == "configured")
            breakdowns.append(CandidateScoreBreakdown(
                model_id=model.model_id,
                eligible=is_eligible,
                ineligible_reason=None if is_eligible else "Model unavailable or unconfigured.",
                candidate_score=predicted_score if is_proposed else 0.0
            ))

        reasoning = [
            f"RL Contextual Bandit Policy evaluated 12-dimensional state vector.",
            f"Proposed shadow candidate '{proposed_id}' with predicted expected reward {predicted_score:.4f}.",
            "Production routing remains controlled by BaselineAdaptivePolicy."
        ]

        return proposed_id, predicted_score, breakdowns, reasoning
