from typing import List, Dict, Any

class ReliabilityMetricsEvaluator:
    """
    Reliability & Failure Analysis Evaluator.
    Calculates Success Rate, Timeout Rate, Provider 429 Quota Rate, Failover Rate, and Error Rate.
    """

    @staticmethod
    def evaluate_reliability(sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sample_results:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "timeout_rate": 0.0,
                "quota_failure_rate": 0.0,
                "failover_rate": 0.0,
                "error_rate": 0.0
            }

        total = len(sample_results)
        successes = sum(1 for r in sample_results if r.get("success", False))
        timeouts = sum(1 for r in sample_results if "timeout" in str(r.get("error_message", "")).lower())
        quotas = sum(1 for r in sample_results if "quota" in str(r.get("error_message", "")).lower() or r.get("failover_used", False))
        failovers = sum(1 for r in sample_results if r.get("failover_used", False))
        errors = total - successes

        return {
            "total_requests": total,
            "successful_requests": successes,
            "failed_requests": errors,
            "success_rate": round(successes / total, 4),
            "timeout_rate": round(timeouts / total, 4),
            "quota_failure_rate": round(quotas / total, 4),
            "failover_rate": round(failovers / total, 4),
            "error_rate": round(errors / total, 4)
        }
