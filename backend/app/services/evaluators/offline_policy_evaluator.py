import logging
import numpy as np
from typing import List, Dict, Any, Optional

from app.schemas.experience import ExperienceRecord
from app.services.policies.base_policy import BaseDecisionPolicy

logger = logging.getLogger("orchestrator")

class OffPolicyEvaluator:
    """
    Phase 6 — Statistically Valid Off-Policy Evaluator for Contextual Bandits.
    Supports Inverse Propensity Scoring (IPS) and Self-Normalized IPS (SNIPS).
    """

    def __init__(self):
        pass

    def evaluate_policy_ips(
        self,
        records: List[ExperienceRecord],
        target_policy_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Computes IPS and SNIPS policy value estimates over propensity-eligible experience records.
        """
        total_records = len(records)
        usable_records = []
        weights = []
        rewards = []

        for rec, pred in zip(records, target_policy_predictions):
            # Safeguard 1: Reject missing or invalid propensity values
            if not getattr(rec, "propensity_available", False):
                continue
            
            p_a = getattr(rec, "propensity_probability", None)
            if p_a is None or p_a <= 0.0 or p_a > 1.0:
                continue

            behavior_action = rec.action if rec.action is not None else rec.behavior_action
            target_action = pred.get("action_index")

            if behavior_action is None or target_action is None or target_action < 0:
                continue

            r_i = float(rec.reward)
            
            # Indicator function: 1.0 if target policy selected same action as behavior policy
            indicator = 1.0 if (target_action == behavior_action) else 0.0
            w_i = indicator / p_a

            usable_records.append(rec)
            weights.append(w_i)
            rewards.append(r_i)

        n_usable = len(usable_records)

        if n_usable == 0:
            logger.info("No usable IPS records found with valid online propensity probabilities P(a|s).")
            return {
                "total_records_inspected": total_records,
                "usable_ips_records_count": 0,
                "ips_valid": False,
                "ips_value": None,
                "snips_value": None,
                "effective_sample_size": 0.0,
                "reason": "Historical propensity scores were not logged; therefore valid IPS cannot be computed for those historical records."
            }

        weights_arr = np.array(weights, dtype=np.float64)
        rewards_arr = np.array(rewards, dtype=np.float64)

        # IPS Value Estimate
        v_ips = float(np.mean(weights_arr * rewards_arr))

        # SNIPS Value Estimate
        sum_w = float(np.sum(weights_arr))
        if sum_w > 0.0:
            v_snips = float(np.sum(weights_arr * rewards_arr) / sum_w)
        else:
            v_snips = 0.0

        # Effective Sample Size: N_eff = (sum w_i)^2 / sum (w_i^2)
        sum_w_sq = float(np.sum(weights_arr ** 2))
        n_eff = float((sum_w ** 2) / sum_w_sq) if sum_w_sq > 0.0 else 0.0

        return {
            "total_records_inspected": total_records,
            "usable_ips_records_count": n_usable,
            "ips_valid": True,
            "ips_value": round(v_ips, 4),
            "snips_value": round(v_snips, 4),
            "effective_sample_size": round(n_eff, 2),
            "mean_importance_weight": round(float(np.mean(weights_arr)), 4),
            "max_importance_weight": round(float(np.max(weights_arr)), 4)
        }
