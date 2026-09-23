import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from app.services.policies.base_policy import BaseDecisionPolicy
from app.schemas.model import ModelMetadata
from app.schemas.decision import CandidateScoreBreakdown
from app.schemas.rl import ShadowDecision
from app.services.experience_buffer import ACTION_MAP, REVERSE_ACTION_MAP, INTENT_MAP

logger = logging.getLogger("orchestrator")

class RLContextualBanditPolicy(BaseDecisionPolicy):
    _instance: Optional["RLContextualBanditPolicy"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RLContextualBanditPolicy, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        if model_path is not None:
            raw_path = model_path
            self.model_path = raw_path if os.path.isabs(raw_path) else os.path.join(project_root, raw_path)
            self._initialized = False

        if getattr(self, "_initialized", False):
            return

        self.state_dim = 12
        self.action_dim = 8
        self.action_map = ACTION_MAP
        self.reverse_action_map = REVERSE_ACTION_MAP
        
        if not hasattr(self, "model_path") or not self.model_path:
            raw_path = os.path.join("data", "rl", "models", "rl_contextual_bandit_policy.json")
            self.model_path = raw_path if os.path.isabs(raw_path) else os.path.join(project_root, raw_path)
        
        self.weights: Optional[np.ndarray] = None  # Shape (8, 12)
        self.bias: Optional[np.ndarray] = None     # Shape (8,)
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
            logger.error(f"RL Policy file not found at '{self.model_path}'. RL routing is unavailable; baseline fallback is required.")
            self._mark_unavailable()
            return False

        try:
            with open(self.model_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("policy_name") != "rl_contextual_bandit_policy":
                logger.error("RL Policy name is missing or invalid.")
                self._mark_unavailable()
                return False

            version = data.get("version")
            if not isinstance(version, str) or not version or not version[0].isdigit():
                logger.error("RL Policy version is missing or invalid.")
                self._mark_unavailable()
                return False

            if data.get("state_dim") != 12:
                logger.error(f"RL Policy state_dim mismatch ({data.get('state_dim')} != 12).")
                self._mark_unavailable()
                return False

            weights_list = data.get("weights")
            bias_list = data.get("bias")
            if weights_list is None or bias_list is None:
                logger.error("RL Policy file missing weights or bias arrays.")
                self._mark_unavailable()
                return False

            if data.get("action_map") != self.action_map:
                logger.error("RL Policy action_map is missing or inconsistent with the authoritative runtime registry.")
                self._mark_unavailable()
                return False

            w_arr = np.array(weights_list, dtype=np.float32)
            b_arr = np.array(bias_list, dtype=np.float32)

            if w_arr.shape != (self.action_dim, self.state_dim) or b_arr.shape != (self.action_dim,):
                logger.error(
                    f"RL Policy dimensions are invalid: weights={w_arr.shape}, bias={b_arr.shape}; "
                    f"expected weights=({self.action_dim}, {self.state_dim}), bias=({self.action_dim},)."
                )
                self._mark_unavailable()
                return False

            self.weights = w_arr
            self.bias = b_arr

            self.policy_version = data.get("version", "2.0.0")
            self.training_samples_count = data.get("training_samples_count", 0)
            self.mean_squared_error = data.get("mean_squared_error")
            self.last_trained_timestamp = data.get("timestamp")

            logger.info(f"Loaded RL Contextual Bandit Policy v{self.policy_version} (weights shape: {self.weights.shape}) from '{self.model_path}'")
            return True
        except Exception as e:
            logger.error(f"Failed to load RL Contextual Bandit Policy from '{self.model_path}': {str(e)}")
            self._mark_unavailable()
            return False

    def _mark_unavailable(self):
        self.weights = None
        self.bias = None
        self.policy_version = None
        self.training_samples_count = 0
        self.mean_squared_error = None
        self.last_trained_timestamp = None

    def _init_default_weights(self):
        rng = np.random.default_rng(seed=42)
        self.weights = 0.01 * rng.standard_normal((self.action_dim, self.state_dim), dtype=np.float32)
        self.bias = np.zeros((self.action_dim,), dtype=np.float32)
        self.policy_version = "2.0.0-uninitialized"

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

        cpu_raw = resource_info.get("cpu_utilization_percent") if resource_info else None
        mem_raw = resource_info.get("memory_utilization_percent") if resource_info else None

        cpu_util = float(cpu_raw if cpu_raw is not None else 20.0) / 100.0
        mem_util = float(mem_raw if mem_raw is not None else 50.0) / 100.0
        gpu_avail = 1.0 if (resource_info and resource_info.get("gpu_available")) else 0.0

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

        # Compute Q_hat(s, a) = W_a^T s + b_a for each action index (8 actions)
        raw_scores = np.dot(self.weights, s_vec) + self.bias
        masked_scores = raw_scores.copy()

        action_masked = False
        # Action Masking based on candidate model availability
        for a_idx in range(k_actions):
            model_id = self.reverse_action_map.get(a_idx)
            if not model_id or model_id == "BAAI/bge-m3":
                # Mask embedding models or unmapped action slots
                masked_scores[a_idx] = -float("inf")
                action_masked = True
                continue

            cand = next((m for m in candidate_models if m.model_id == model_id), None)
            if not cand or not cand.available or cand.configuration_status != "configured":
                masked_scores[a_idx] = -float("inf")
                action_masked = True

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
        state_vec = self.encode_state_vector(intent_info, complexity_info, resource_info)
        shadow_dec = self.predict_shadow_decision(state_vec, None, candidate_models)

        breakdowns: List[CandidateScoreBreakdown] = []

        if not shadow_dec.policy_available or not shadow_dec.proposed_model:
            reasoning = [
                f"RL Policy '{self.policy_name}' evaluated 12-dimensional state vector.",
                "Policy weights unavailable or all actions masked.",
                "Fallback to BaselineAdaptivePolicy required."
            ]
            for model in candidate_models:
                breakdowns.append(CandidateScoreBreakdown(
                    model_id=model.model_id,
                    eligible=False,
                    ineligible_reason="RL Action masked or unavailable.",
                    candidate_score=0.0
                ))
            return None, 0.0, breakdowns, reasoning

        proposed_id = shadow_dec.proposed_model
        predicted_score = shadow_dec.predicted_reward

        # Action Validation Gate: Verify proposed_id is not an embedding model
        if proposed_id == "BAAI/bge-m3":
            logger.warning(f"RL Policy selected embedding model '{proposed_id}'. Masking action and triggering fallback.")
            return None, 0.0, [], ["Embedding model forbidden for text generation."]

        for model in candidate_models:
            is_proposed = (model.model_id == proposed_id)
            is_eligible = (model.model_type == "llm" and model.available and model.configuration_status == "configured" and model.model_id != "BAAI/bge-m3")
            breakdowns.append(CandidateScoreBreakdown(
                model_id=model.model_id,
                provider=model.provider,
                display_name=model.display_name or model.model_id,
                eligible=is_eligible,
                ineligible_reason=None if is_eligible else "Model unavailable or unconfigured.",
                candidate_score=predicted_score if is_proposed else 0.0
            ))

        reasoning = [
            f"RL Contextual Bandit Policy v{self.policy_version or '2.0.0'} evaluated 12-dimensional state vector.",
            f"Selected candidate model '{proposed_id}' with expected reward Q(s,a) = {predicted_score:.4f}.",
            "Production routing executed via trained Contextual Bandit Policy."
        ]

        return proposed_id, predicted_score, breakdowns, reasoning
