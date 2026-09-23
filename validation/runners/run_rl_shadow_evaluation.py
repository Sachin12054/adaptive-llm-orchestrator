import os
import sys
import json
import logging
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.strategies.rl_shadow_policy_adapter import RLShadowPolicyAdapter
from validation.metrics.statistical_metrics import StatisticalMetricsEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validation")

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "statistical"))

def run_rl_shadow_evaluation() -> Dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, "rl_shadow_evaluation_results.json")

    logger.info("Executing Offline RL Shadow Evaluation (Strategy H)...")
    adapter = RLShadowPolicyAdapter()
    res = adapter.evaluate_offline_buffer()

    logger.info(f"Offline RL Evaluation Complete: IPS={res.get('estimated_policy_value_ips')}, SNIPS={res.get('estimated_policy_value_snips')}, ESS={res.get('effective_sample_size_ess')}, Agreement={res.get('policy_agreement_rate')}")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res

if __name__ == "__main__":
    run_rl_shadow_evaluation()
