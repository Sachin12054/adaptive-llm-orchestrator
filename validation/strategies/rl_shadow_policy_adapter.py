import os
import json
import logging
from typing import Dict, Any, List

from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.services.experience_buffer import ExperienceBufferService

logger = logging.getLogger("validation")

class RLShadowPolicyAdapter:
    """
    Strategy H: RL Contextual Bandit Policy (Shadow Evaluator).
    IMPORTANT: Evaluated strictly OFFLINE on propensity-logged experience buffer data.
    Never used for live production routing!
    """

    def __init__(self, rl_policy: RLContextualBanditPolicy = None, buffer: ExperienceBufferService = None):
        self.rl_policy = rl_policy or RLContextualBanditPolicy()
        self.buffer = buffer or ExperienceBufferService()

    def evaluate_offline_buffer(self) -> Dict[str, Any]:
        """
        Evaluates the RL policy offline against propensity-logged historical records.
        Calculates Policy Agreement, IPS, SNIPS, ESS, Positivity, and Action Distributions.
        """
        records = list(self.buffer._buffer)
        if not records:
            return {
                "status": "INSUFFICIENT_DATA",
                "sample_count": 0,
                "message": "No experience records available in buffer."
            }

        valid_records = [
            r for r in records
            if getattr(r, "propensity_probability", None) is not None and r.propensity_probability > 0.0
        ]
        total_count = len(records)
        valid_count = len(valid_records)

        agreed_count = 0
        rewards = []
        propensities = []
        rl_actions = []
        baseline_actions = []

        weights_ips = []
        weights_snips_numerators = []

        import numpy as np

        for r in valid_records:
            state = r.state if (hasattr(r, "state") and r.state and len(r.state) == 12) else [0.5]*12
            s_vec = np.array(state, dtype=np.float32)
            
            if self.rl_policy.weights is not None and self.rl_policy.bias is not None:
                q_scores = np.dot(self.rl_policy.weights, s_vec) + self.rl_policy.bias
                # Mask Action 7 (BAAI/bge-m3)
                q_scores[7] = -float("inf")
                action_idx = int(np.argmax(q_scores))
            else:
                action_idx = 0

            baseline_act = getattr(r, "action_model_id", None) or getattr(r, "behavior_model_id", "gemma-3-4b")
            rl_act_model = self.rl_policy.reverse_action_map.get(action_idx, "gemma-3-4b")
            
            # Mask Action 7 (BAAI/bge-m3)
            if rl_act_model == "BAAI/bge-m3":
                rl_act_model = "gemma-3-4b"

            rl_actions.append(rl_act_model)
            baseline_actions.append(baseline_act)

            prop = r.propensity_probability
            if rl_act_model == baseline_act:
                agreed_count += 1
                w = 1.0 / max(prop, 0.01)
                weights_ips.append(w * r.reward)
                weights_snips_numerators.append(w)
                rewards.append(r.reward)
            propensities.append(prop)

        agreement_rate = agreed_count / max(valid_count, 1)
        ips_val = sum(weights_ips) / max(valid_count, 1) if valid_count > 0 else 0.0
        snips_denom = sum(weights_snips_numerators)
        snips_val = sum(weights_ips) / max(snips_denom, 1e-6) if snips_denom > 0 else 0.0

        # Effective Sample Size (ESS) = (sum w_i)^2 / sum (w_i^2)
        ess = (snips_denom ** 2) / max(sum(w**2 for w in weights_snips_numerators), 1e-6) if weights_snips_numerators else 0.0

        return {
            "status": "SUCCESS",
            "total_records": total_count,
            "valid_propensity_records": valid_count,
            "agreed_records": agreed_count,
            "policy_agreement_rate": round(agreement_rate, 4),
            "estimated_policy_value_ips": round(ips_val, 4),
            "estimated_policy_value_snips": round(snips_val, 4),
            "effective_sample_size_ess": round(ess, 2),
            "positivity_coverage": round(valid_count / max(total_count, 1), 4),
            "mean_observed_reward": round(sum(rewards)/max(len(rewards), 1), 4) if rewards else 0.0,
            "rl_action_distribution": {m: rl_actions.count(m) for m in set(rl_actions)}
        }
