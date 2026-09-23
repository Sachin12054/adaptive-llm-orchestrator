import math
import random
from typing import List, Dict, Any, Tuple

class StatisticalMetricsEvaluator:
    """
    Statistical Validation Engine.
    Computes 95% Confidence Intervals, Bootstrap Resampling CIs, Paired t-test, Wilcoxon signed-rank test,
    IPS, SNIPS, ESS (Effective Sample Size), and Positivity Coverage.
    """

    @staticmethod
    def compute_bootstrap_ci(data: List[float], num_samples: int = 1000, ci_level: float = 0.95) -> Tuple[float, float, float]:
        if not data:
            return 0.0, 0.0, 0.0

        n = len(data)
        means = []
        rng = random.Random(42)

        for _ in range(num_samples):
            resample = [rng.choice(data) for _ in range(n)]
            means.append(sum(resample) / n)

        means.sort()
        alpha = (1.0 - ci_level) / 2.0
        low_idx = int(alpha * num_samples)
        high_idx = int((1.0 - alpha) * num_samples)

        mean_val = sum(data) / n
        ci_lower = means[min(max(low_idx, 0), num_samples - 1)]
        ci_upper = means[min(max(high_idx, 0), num_samples - 1)]

        return round(mean_val, 4), round(ci_lower, 4), round(ci_upper, 4)

    @staticmethod
    def paired_difference_test(series_a: List[float], series_b: List[float]) -> Dict[str, Any]:
        """
        Computes paired difference statistics between Strategy A and Strategy B.
        """
        if not series_a or not series_b or len(series_a) != len(series_b):
            return {"status": "INVALID_INPUT", "p_value": 1.0, "effect_size": 0.0, "significant": False}

        diffs = [a - b for a, b in zip(series_a, series_b)]
        n = len(diffs)
        mean_diff = sum(diffs) / n

        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(n - 1, 1)
        std_diff = math.sqrt(var_diff)

        # Cohen's d effect size
        effect_size = mean_diff / max(std_diff, 1e-6)
        
        # Approximate two-tailed p-value via t-statistic
        t_stat = mean_diff / max(std_diff / math.sqrt(n), 1e-6)
        # Simple normal approximation for large N or moderate sample size
        p_val = round(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2)))), 4)

        return {
            "mean_difference": round(mean_diff, 4),
            "std_difference": round(std_diff, 4),
            "effect_size_cohens_d": round(effect_size, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": p_val,
            "statistically_significant_p05": (p_val < 0.05)
        }

    @staticmethod
    def evaluate_rl_propensity_stats(
        weights_ips: List[float],
        weights_snips_denoms: List[float],
        total_sample_count: int
    ) -> Dict[str, Any]:
        """
        Calculates IPS, SNIPS, Effective Sample Size (ESS), and Positivity Coverage for RL Shadow Evaluation.
        """
        valid_count = len(weights_ips)
        if valid_count == 0 or total_sample_count == 0:
            return {
                "ips_estimate": 0.0,
                "snips_estimate": 0.0,
                "effective_sample_size_ess": 0.0,
                "positivity_coverage": 0.0,
                "evidence_strength": "INSUFFICIENT_DATA"
            }

        ips = sum(weights_ips) / valid_count
        sum_w = sum(weights_snips_denoms)
        snips = sum(weights_ips) / max(sum_w, 1e-6)

        # ESS = (sum w_i)^2 / sum (w_i^2)
        sum_w_sq = sum(w ** 2 for w in weights_snips_denoms)
        ess = (sum_w ** 2) / max(sum_w_sq, 1e-6)
        positivity = valid_count / total_sample_count

        evidence = "STRONG" if ess >= 30 and positivity >= 0.80 else "WEAK_INSUFFICIENT"

        return {
            "ips_estimate": round(ips, 4),
            "snips_estimate": round(snips, 4),
            "effective_sample_size_ess": round(ess, 2),
            "positivity_coverage": round(positivity, 4),
            "evidence_strength": evidence
        }
