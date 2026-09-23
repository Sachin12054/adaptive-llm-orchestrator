import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from validation.metrics.quality_metrics import QualityMetricsEvaluator
from validation.metrics.latency_metrics import LatencyMetricsEvaluator
from validation.metrics.cost_metrics import CostMetricsEvaluator
from validation.metrics.statistical_metrics import StatisticalMetricsEvaluator

def test_dataset_count_and_categories():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "benchmark_dataset.jsonl"))
    assert os.path.exists(dataset_path), "Dataset jsonl file missing!"

    prompts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    assert len(prompts) == 113, f"Expected 113 prompts, found {len(prompts)}"
    
    categories = set(p["category"] for p in prompts)
    assert len(categories) == 15, f"Expected 15 categories, found {len(categories)}"

def test_safety_invariants_strictly_enforced():
    assert getattr(settings, "PRODUCTION_OVERRIDE", False) is False, "Safety Invariant Violated: PRODUCTION_OVERRIDE is True!"

def test_quality_metrics_evaluation_rubric():
    res = QualityMetricsEvaluator.evaluate_quality(
        prompt="Design a campus database structure for students",
        category="SQL / Database",
        generated_text="Here is the table structure: Students (id, name, email), Courses (course_id, title).",
        success=True
    )
    assert 0.0 <= res["score_0_to_5"] <= 5.0
    assert "Good" in res["rubric_level"] or "Excellent" in res["rubric_level"] or "Acceptable" in res["rubric_level"]

def test_latency_metrics_percentiles_and_speedup():
    lats = [10.0, 20.0, 30.0, 40.0, 50.0]
    p_stats = LatencyMetricsEvaluator.compute_percentiles(lats)
    assert p_stats["mean"] == 30.0
    assert p_stats["p50"] == 30.0
    
    speedup = LatencyMetricsEvaluator.calculate_parallel_speedup(100.0, 50.0)
    assert speedup == 2.0

def test_cost_metrics_calculation():
    cost_res = CostMetricsEvaluator.calculate_cost("Google Gemini API", 1000, 2000)
    assert cost_res["input_cost_usd"] == 0.000075
    assert cost_res["output_cost_usd"] == 0.000600
    assert cost_res["total_cost_usd"] == 0.000675

def test_statistical_metrics_bootstrap_and_rl_propensity():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, ci_low, ci_high = StatisticalMetricsEvaluator.compute_bootstrap_ci(data)
    assert ci_low <= mean <= ci_high

    rl_stats = StatisticalMetricsEvaluator.evaluate_rl_propensity_stats(
        weights_ips=[1.5, 2.0, 1.8],
        weights_snips_denoms=[1.0, 1.2, 0.9],
        total_sample_count=10
    )
    assert rl_stats["positivity_coverage"] == 0.3
    assert rl_stats["effective_sample_size_ess"] > 0.0
