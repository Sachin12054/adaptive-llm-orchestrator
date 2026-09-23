import os
import sys
import json
import time
import logging
from typing import Dict, List, Any

# Ensure backend package is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_path = os.path.join(project_root, "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluation_runner")

ESTIMATED_COST_PER_1K: Dict[str, float] = {
    "gemma-3-4b": 0.0,
    "qwen-coder-3b": 0.0,
    "deepseek-r1-7b": 0.0,
    "gemini-3.5-flash": 0.00015,
    "mistral-small-latest": 0.00020,
    "llama-3.3-70b-versatile": 0.00059,
    "meta-llama/llama-3.3-70b-instruct": 0.00040,
}

def estimate_cost(model_id: str, prompt_text: str, generated_text: str) -> float:
    if not model_id:
        return 0.0
    rate = ESTIMATED_COST_PER_1K.get(model_id.lower(), 0.0)
    if rate == 0.0:
        return 0.0
    prompt_words = len(prompt_text.split()) if prompt_text else 0
    output_words = len(generated_text.split()) if generated_text else 0
    est_tokens = (prompt_words + output_words) * 1.33
    return round((est_tokens / 1000.0) * rate, 6)

def run_benchmark():
    dataset_path = os.path.join(project_root, "datasets", "evaluation", "benchmark_100.json")
    results_dir = os.path.join(project_root, "data", "evaluation")
    results_path = os.path.join(results_dir, "benchmark_100_results.jsonl")

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
    total_latency_ms = 0.0
    total_reward = 0.0
    total_cost = 0.0

    intent_correct = 0
    complexity_correct = 0
    fallback_count = 0
    rl_agreement_count = 0

    model_distribution: Dict[str, int] = {}
    complexity_distribution: Dict[str, int] = {}

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
                # Correct key extraction: 'level' or 'complexity_level'
                detected_complexity = complexity_info.get("level") or complexity_info.get("complexity_level")
                complexity_score = complexity_info.get("complexity_score", 0.0)

                selected_model = resp.selected_model or gen_data.get("model_id") or "none"
                provider = gen_data.get("provider") or "none"
                exec_success = resp.success and gen_data.get("success", False)
                gen_text = gen_data.get("generated_text") or ""

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

                if detected_intent == expected_intent:
                    intent_correct += 1
                if detected_complexity == expected_complexity:
                    complexity_correct += 1
                if fallback_triggered:
                    fallback_count += 1
                if rl_proposed and rl_proposed == selected_model:
                    rl_agreement_count += 1

                model_distribution[selected_model] = model_distribution.get(selected_model, 0) + 1
                complexity_distribution[detected_complexity or "unknown"] = complexity_distribution.get(detected_complexity or "unknown", 0) + 1

                total_latency_ms += resp.pipeline_latency_ms
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
                    "fallback": fallback_triggered,
                    "rl_proposed_model": rl_proposed,
                    "rl_production_override": rl_override
                }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
            else:
                failed_executions += 1
                lat = round((t_end - t_start) * 1000, 2)
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
                    "execution_latency_ms": lat,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost": 0.0,
                    "resource_telemetry": {},
                    "reward": 0.0,
                    "execution_success": False,
                    "fallback": False,
                    "rl_proposed_model": None,
                    "rl_production_override": False,
                    "error_message": last_error or "Pipeline execution error"
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

    end_all_time = time.perf_counter()
    total_wall_time_sec = round(end_all_time - start_all_time, 2)

    avg_latency = round(total_latency_ms / total_prompts, 2) if total_prompts > 0 else 0.0
    avg_reward = round(total_reward / total_prompts, 4) if total_prompts > 0 else 0.0
    avg_cost = round(total_cost / total_prompts, 6) if total_prompts > 0 else 0.0
    intent_acc = round((intent_correct / total_prompts) * 100, 2) if total_prompts > 0 else 0.0
    cmplx_acc = round((complexity_correct / total_prompts) * 100, 2) if total_prompts > 0 else 0.0
    rl_agreement_rate = round((rl_agreement_count / total_prompts) * 100, 2) if total_prompts > 0 else 0.0

    summary = {
        "total_prompts": total_prompts,
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
        "total_wall_clock_time_sec": total_wall_time_sec,
        "average_latency_ms": avg_latency,
        "average_reward": avg_reward,
        "average_cost_usd": avg_cost,
        "routing_distribution_by_model": model_distribution,
        "routing_distribution_by_complexity": complexity_distribution,
        "intent_classification_accuracy_percent": intent_acc,
        "complexity_classification_accuracy_percent": cmplx_acc,
        "fallback_count": fallback_count,
        "rl_baseline_agreement_rate_percent": rl_agreement_rate
    }

    summary_path = os.path.join(results_dir, "benchmark_100_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY REPORT:")
    logger.info(json.dumps(summary, indent=2))
    logger.info("=" * 60)
    logger.info(f"Results written to: {results_path}")
    logger.info(f"Summary written to: {summary_path}")

if __name__ == "__main__":
    run_benchmark()
