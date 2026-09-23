import math
from typing import List, Dict, Any

class LatencyMetricsEvaluator:
    """
    Latency Evaluation & Percentile Statistics (P50, P95, Mean, Std).
    Calculates parallel speedup accurately: controlled_sequential_wall_clock / controlled_parallel_wall_clock.
    """

    @staticmethod
    def compute_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
        if not latencies_ms:
            return {"mean": 0.0, "median": 0.0, "p50": 0.0, "p95": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        sorted_l = sorted(latencies_ms)
        n = len(sorted_l)
        mean_val = sum(sorted_l) / n

        # Standard deviation
        variance = sum((x - mean_val) ** 2 for x in sorted_l) / max(n - 1, 1)
        std_val = math.sqrt(variance)

        def percentile(p: float) -> float:
            idx = int(round(p * (n - 1)))
            return sorted_l[min(max(idx, 0), n - 1)]

        return {
            "mean": round(mean_val, 2),
            "median": round(percentile(0.50), 2),
            "p50": round(percentile(0.50), 2),
            "p95": round(percentile(0.95), 2),
            "std": round(std_val, 2),
            "min": round(sorted_l[0], 2),
            "max": round(sorted_l[-1], 2)
        }

    @staticmethod
    def calculate_parallel_speedup(
        controlled_sequential_wall_clock_ms: float,
        controlled_parallel_wall_clock_ms: float
    ) -> float:
        if controlled_parallel_wall_clock_ms <= 0.0:
            return 1.0
        return round(controlled_sequential_wall_clock_ms / controlled_parallel_wall_clock_ms, 2)
