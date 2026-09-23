import os
import sys
import json
import logging
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.metrics.latency_metrics import LatencyMetricsEvaluator
from validation.metrics.reliability_metrics import ReliabilityMetricsEvaluator
from validation.metrics.statistical_metrics import StatisticalMetricsEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validation")

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "processed"))
STATISTICAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "statistical"))
SUMMARY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "final_validation_summary.json"))
MANIFEST_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "experiment_manifest.json"))

STRATEGIES = ["A", "B", "C", "D", "E", "F", "G"]

def process_raw_strategy_results() -> Dict[str, Any]:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(STATISTICAL_DIR, exist_ok=True)

    summary_table = []
    manifest_records = []

    baseline_g_scores = []
    baseline_g_latencies = []

    for strat_id in STRATEGIES:
        raw_file = os.path.join(RAW_DIR, f"raw_results_strategy_{strat_id}.jsonl")
        if not os.path.exists(raw_file):
            logger.warning(f"Raw results for Strategy {strat_id} not found: {raw_file}")
            summary_table.append({
                "strategy_id": strat_id,
                "status": "NOT MEASURED",
                "reason": "Execution raw file unavailable"
            })
            continue

        records = []
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if not records:
            continue

        latencies = [r["total_latency_ms"] for r in records]
        scores_5 = [r["score_0_to_5"] for r in records]
        costs = [r["total_cost_usd"] for r in records]

        if strat_id == "G":
            baseline_g_scores = scores_5
            baseline_g_latencies = latencies

        lat_stats = LatencyMetricsEvaluator.compute_percentiles(latencies)
        rel_stats = ReliabilityMetricsEvaluator.evaluate_reliability(records)

        mean_q, q_ci_low, q_ci_high = StatisticalMetricsEvaluator.compute_bootstrap_ci(scores_5)
        mean_l, l_ci_low, l_ci_high = StatisticalMetricsEvaluator.compute_bootstrap_ci(latencies)

        total_cost = sum(costs)
        avg_cost = total_cost / len(costs)

        strat_summary = {
            "strategy_id": strat_id,
            "strategy_name": records[0].get("strategy", f"Strategy_{strat_id}"),
            "sample_count": len(records),
            "quality_mean_0_to_5": round(mean_q, 2),
            "quality_95_ci": [q_ci_low, q_ci_high],
            "latency_mean_ms": lat_stats["mean"],
            "latency_p50_ms": lat_stats["p50"],
            "latency_p95_ms": lat_stats["p95"],
            "latency_std_ms": lat_stats["std"],
            "latency_95_ci": [l_ci_low, l_ci_high],
            "total_cost_usd": round(total_cost, 4),
            "average_cost_usd": round(avg_cost, 6),
            "success_rate": rel_stats["success_rate"],
            "failover_rate": rel_stats["failover_rate"],
            "status": "PRIMARY_CONTROLLED_MEASURED"
        }

        summary_table.append(strat_summary)

        # Save processed JSON
        with open(os.path.join(PROCESSED_DIR, f"summary_strategy_{strat_id}.json"), "w", encoding="utf-8") as f:
            json.dump(strat_summary, f, indent=2)

        manifest_records.append({
            "experiment_id": f"EXP_BENCHMARK_V1_{strat_id}",
            "strategy": strat_id,
            "dataset_version": "v1.0.0",
            "timestamp": records[-1].get("timestamp", ""),
            "sample_count": len(records),
            "status": "COMPLETED",
            "output_file": raw_file
        })

    # Read Offline RL Shadow Results if available
    rl_shadow_file = os.path.join(STATISTICAL_DIR, "rl_shadow_evaluation_results.json")
    if os.path.exists(rl_shadow_file):
        with open(rl_shadow_file, "r", encoding="utf-8") as f:
            rl_data = json.load(f)

        summary_table.append({
            "strategy_id": "H",
            "strategy_name": "RL Contextual Bandit Policy (Shadow)",
            "sample_count": rl_data.get("valid_propensity_records", 0),
            "estimated_ips_policy_value": rl_data.get("estimated_policy_value_ips", 0.0),
            "estimated_snips_policy_value": rl_data.get("estimated_policy_value_snips", 0.0),
            "effective_sample_size_ess": rl_data.get("effective_sample_size_ess", 0.0),
            "policy_agreement_rate": rl_data.get("policy_agreement_rate", 0.0),
            "positivity_coverage": rl_data.get("positivity_coverage", 0.0),
            "status": "OFFLINE_SHADOW_ESTIMATE_ONLY"
        })

    # Save final summary JSON
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)

    # Save manifest JSON
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

    logger.info(f"Saved final validation summary to {SUMMARY_FILE}")
    logger.info(f"Saved experiment manifest to {MANIFEST_FILE}")

    return {"summary_table": summary_table, "manifest": manifest_records}

if __name__ == "__main__":
    process_raw_strategy_results()
