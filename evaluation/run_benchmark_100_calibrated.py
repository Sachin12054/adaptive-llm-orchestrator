import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any
from collections import Counter

# Ensure backend package is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_path = os.path.join(project_root, "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Ensure sys.stdout buffers are flushed immediately
sys.stdout.reconfigure(line_buffering=True)

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrated_evaluation_runner")

ESTIMATED_COST_PER_1K: Dict[str, float] = {
    "gemma-3-4b": 0.0,
    "qwen-coder-3b": 0.0,
    "deepseek-r1-7b": 0.0,
    "gemini-3.5-flash": 0.00015,
    "mistral-small-latest": 0.00020,
    "llama-3.3-70b-versatile": 0.00059,
    "meta-llama/llama-3.3-70b-instruct": 0.00040,
}

MODEL_CAPABILITY_TIER: Dict[str, str] = {
    "gemma-3-4b": "Small/Local",
    "qwen-coder-3b": "Small/Local",
    "deepseek-r1-7b": "Small/Local",
    "gemini-3.5-flash": "Medium",
    "mistral-small-latest": "Medium",
    "llama-3.3-70b-versatile": "Large/70B",
    "meta-llama/llama-3.3-70b-instruct": "Large/70B",
    "none": "Unknown"
}

def estimate_cost(model_id: str, prompt_text: str, generated_text: str) -> float:
    if not model_id or model_id == "none":
        return 0.0
    rate = ESTIMATED_COST_PER_1K.get(model_id.lower(), 0.0)
    if rate == 0.0:
        return 0.0
    prompt_words = len(prompt_text.split()) if prompt_text else 0
    output_words = len(generated_text.split()) if generated_text else 0
    est_tokens = (prompt_words + output_words) * 1.33
    return round((est_tokens / 1000.0) * rate, 6)

def classify_error(error_msg: str, status_code: int = 0) -> str:
    if not error_msg:
        return "none"
    err_lower = error_msg.lower()
    if "429" in err_lower or "quota" in err_lower or "rate limit" in err_lower or "resource_exhausted" in err_lower:
        return "rate_limit_failure"
    elif "500" in err_lower or "503" in err_lower or "502" in err_lower or "api key" in err_lower or "unauthorized" in err_lower:
        return "provider_api_failure"
    elif "routing" in err_lower or "candidate" in err_lower:
        return "routing_failure"
    elif "validation" in err_lower or "empty response" in err_lower:
        return "validation_response_failure"
    elif "connect" in err_lower or "timeout" in err_lower:
        return "infrastructure_failure"
    else:
        return "model_execution_failure"

def run_calibrated_benchmark():
    dataset_path = os.path.join(project_root, "datasets", "evaluation", "benchmark_100.json")
    results_dir = os.path.join(project_root, "data", "evaluation")
    
    results_path = os.path.join(results_dir, "benchmark_100_calibrated_results.jsonl")
    summary_path = os.path.join(results_dir, "benchmark_100_calibrated_summary.json")

    old_summary_path = os.path.join(results_dir, "benchmark_100_summary.json")
    old_summary = {}
    if os.path.exists(old_summary_path):
        with open(old_summary_path, "r", encoding="utf-8") as f:
            old_summary = json.load(f)

    if not os.path.exists(dataset_path):
        logger.error(f"Benchmark dataset file not found at '{dataset_path}'")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmarks: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(benchmarks)} prompts from '{dataset_path}'")

    pipeline = OrchestrationPipeline()
    # Warmup singleton prototype services
    pipeline.decision_engine.intent_classifier.prototype_service.get_prototype_embeddings()
    pipeline.decision_engine.complexity_analyzer.prototype_service.get_prototype_embeddings()

    total_prompts = len(benchmarks)
    successful_executions = 0
    failed_executions = 0
    total_reward = 0.0
    total_cost = 0.0

    intent_correct = 0
    complexity_correct = 0
    fallback_count = 0
    rl_agreement_count = 0

    latencies: List[float] = []
    rewards_by_tier = {"low": [], "medium": [], "high": [], "very_high": []}
    costs_by_tier = {"low": [], "medium": [], "high": [], "very_high": []}
    
    model_distribution: Dict[str, int] = {}
    complexity_distribution: Dict[str, int] = {}
    provider_distribution: Dict[str, int] = {}
    error_categories: Dict[str, int] = Counter()

    # Routing matrices: Ground-truth tier vs Model / Provider / Capability tier
    routing_by_gt_tier: Dict[str, Counter] = {t: Counter() for t in ["low", "medium", "high", "very_high"]}
    capability_matrix: Dict[str, Counter] = {t: Counter() for t in ["low", "medium", "high", "very_high"]}

    records = []

    start_all_time = time.perf_counter()

    with open(results_path, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(benchmarks, 1):
            prompt_id = item["id"]
            prompt_text = item["prompt"]
            expected_intent = item["intent"]
            expected_complexity = item["expected_complexity"]

            logger.info(f"[{idx}/{total_prompts}] Executing {prompt_id} (Expected: {expected_intent} | {expected_complexity})...")

            req = OrchestrationRequest(
                prompt=prompt_text,
                execution_mode="online"
            )

            t_start = time.perf_counter()
            resp = None
            last_error = None

            try:
                resp = pipeline.run_pipeline(req)
            except Exception as e:
                last_error = str(e)
                logger.error(f"Pipeline exception on {prompt_id}: {last_error}")

            t_end = time.perf_counter()
            exec_lat_ms = round((t_end - t_start) * 1000, 2)

            if resp:
                decision_data = resp.decision.dict() if resp.decision else {}
                gen_data = resp.generation.dict() if resp.generation else {}
                ver_data = resp.verification.dict() if resp.verification else {}
                reward_data = resp.reward.dict() if resp.reward else {}

                intent_info = decision_data.get("intent_info") or {}
                complexity_info = decision_data.get("complexity_info") or {}
                resource_info = decision_data.get("resource_info") or {}
                shadow_rl = decision_data.get("shadow_rl_decision") or {}

                detected_intent = intent_info.get("intent")
                detected_complexity = complexity_info.get("level") or complexity_info.get("complexity_level")
                complexity_score = complexity_info.get("complexity_score", 0.0)

                selected_model = resp.selected_model or gen_data.get("model_id") or "none"
                provider = gen_data.get("provider") or "none"
                exec_success = resp.success and gen_data.get("success", False)
                gen_text = gen_data.get("generated_text") or ""
                error_msg = gen_data.get("error_message") or last_error or ""

                err_cat = classify_error(error_msg) if not exec_success else "none"

                usage = gen_data.get("usage") or {}
                input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or len(prompt_text.split()) * 2
                output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or len(gen_text.split()) * 2
                cost = estimate_cost(selected_model, prompt_text, gen_text)

                reward_val = reward_data.get("reward_score", 0.0)
                fallback_triggered = gen_data.get("fallback_triggered", False) or decision_data.get("fallback_triggered", False)

                rl_proposed = shadow_rl.get("proposed_model")
                rl_override = False

                if exec_success:
                    successful_executions += 1
                else:
                    failed_executions += 1
                    error_categories[err_cat] += 1

                if detected_intent == expected_intent:
                    intent_correct += 1
                if detected_complexity == expected_complexity:
                    complexity_correct += 1
                if fallback_triggered:
                    fallback_count += 1
                if rl_proposed and rl_proposed == selected_model:
                    rl_agreement_count += 1

                latencies.append(resp.pipeline_latency_ms)
                rewards_by_tier[expected_complexity].append(reward_val)
                costs_by_tier[expected_complexity].append(cost)

                model_distribution[selected_model] = model_distribution.get(selected_model, 0) + 1
                complexity_distribution[detected_complexity or "unknown"] = complexity_distribution.get(detected_complexity or "unknown", 0) + 1
                provider_distribution[provider] = provider_distribution.get(provider, 0) + 1

                routing_by_gt_tier[expected_complexity][selected_model] += 1
                cap_tier = MODEL_CAPABILITY_TIER.get(selected_model, "Unknown")
                capability_matrix[expected_complexity][cap_tier] += 1

                total_reward += reward_val
                total_cost += cost

                record = {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "expected_intent": expected_intent,
                    "detected_intent": detected_intent,
                    "expected_complexity": expected_complexity,
                    "detected_complexity": detected_complexity,
                    "complexity_score": complexity_score,
                    "selected_model": selected_model,
                    "provider": provider,
                    "execution_latency_ms": resp.pipeline_latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost": round(cost, 6),
                    "resource_telemetry": {
                        "cpu_utilization_percent": resource_info.get("cpu_utilization_percent"),
                        "memory_utilization_percent": resource_info.get("memory_utilization_percent"),
                        "gpu_available": resource_info.get("gpu_available"),
                        "gpu_free_vram_gb": resource_info.get("gpu_free_vram_gb")
                    },
                    "reward": reward_val,
                    "execution_success": exec_success,
                    "error_category": err_cat,
                    "error_message": error_msg if not exec_success else None,
                    "fallback": fallback_triggered,
                    "rl_proposed_model": rl_proposed,
                    "rl_production_override": rl_override
                }

                records.append(record)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

            else:
                failed_executions += 1
                err_cat = classify_error(last_error or "Pipeline execution error")
                error_categories[err_cat] += 1
                latencies.append(exec_lat_ms)

                record = {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "expected_intent": expected_intent,
                    "detected_intent": "error",
                    "expected_complexity": expected_complexity,
                    "detected_complexity": "error",
                    "complexity_score": 0.0,
                    "selected_model": "none",
                    "provider": "none",
                    "execution_latency_ms": exec_lat_ms,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost": 0.0,
                    "resource_telemetry": {},
                    "reward": 0.0,
                    "execution_success": False,
                    "error_category": err_cat,
                    "error_message": last_error or "Pipeline execution error",
                    "fallback": False,
                    "rl_proposed_model": None,
                    "rl_production_override": False
                }
                records.append(record)
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

    end_all_time = time.perf_counter()
    total_wall_time_sec = round(end_all_time - start_all_time, 2)

    avg_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0
    median_latency = round(float(np.median(latencies)), 2) if latencies else 0.0
    p95_latency = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    min_latency = round(float(np.min(latencies)), 2) if latencies else 0.0
    max_latency = round(float(np.max(latencies)), 2) if latencies else 0.0

    avg_reward = round(total_reward / total_prompts, 4) if total_prompts > 0 else 0.0
    avg_cost = round(total_cost / total_prompts, 6) if total_prompts > 0 else 0.0
    intent_acc = round((intent_correct / total_prompts) * 100, 2) if total_prompts > 0 else 0.0
    cmplx_acc = round((complexity_correct / total_prompts) * 100, 2) if total_prompts > 0 else 0.0
    rl_agreement_rate = round((rl_agreement_count / total_prompts) * 100, 2) if total_prompts > 0 else 0.0

    avg_reward_by_tier = {
        tier: round(float(np.mean(rewards_by_tier[tier])), 4) if rewards_by_tier[tier] else 0.0
        for tier in ["low", "medium", "high", "very_high"]
    }
    avg_cost_by_tier = {
        tier: round(float(np.mean(costs_by_tier[tier])), 6) if costs_by_tier[tier] else 0.0
        for tier in ["low", "medium", "high", "very_high"]
    }

    summary = {
        "total_prompts": total_prompts,
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
        "failure_percentage": round((failed_executions / total_prompts) * 100, 2),
        "total_wall_clock_time_sec": total_wall_time_sec,
        "latency_metrics_ms": {
            "average": avg_latency,
            "median": median_latency,
            "p95": p95_latency,
            "min": min_latency,
            "max": max_latency
        },
        "total_cost_usd": round(total_cost, 6),
        "average_cost_usd": avg_cost,
        "average_reward": avg_reward,
        "reward_by_complexity_tier": avg_reward_by_tier,
        "cost_by_complexity_tier": avg_cost_by_tier,
        "routing_distribution_by_model": model_distribution,
        "routing_distribution_by_provider": provider_distribution,
        "routing_distribution_by_complexity": complexity_distribution,
        "error_categories_breakdown": dict(error_categories),
        "intent_classification_accuracy_percent": intent_acc,
        "complexity_classification_accuracy_percent": cmplx_acc,
        "fallback_count": fallback_count,
        "rl_baseline_agreement_rate_percent": rl_agreement_rate,
        "routing_matrix_gt_tier_vs_model": {t: dict(cnt) for t, cnt in routing_by_gt_tier.items()},
        "capability_matrix_gt_tier_vs_capability": {t: dict(cnt) for t, cnt in capability_matrix.items()}
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary JSON written to 'benchmark_100_calibrated_summary.json'")

    # Save detailed validation report JSON
    validation_report = {
        "summary": summary,
        "before_vs_after_comparison": {
            "before": old_summary,
            "after": summary
        },
        "records": records
    }

    val_report_path = os.path.join(project_root, "scratch", "end_to_end_validation_report.json")
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    logger.info(f"End-to-end validation report written to '{val_report_path}'")

if __name__ == "__main__":
    run_calibrated_benchmark()
