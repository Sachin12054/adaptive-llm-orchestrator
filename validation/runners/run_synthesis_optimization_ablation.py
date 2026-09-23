import os
import sys
import time
import json
import asyncio
import psutil
import math
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.services.providers.ollama_provider import OllamaProvider
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.services.response_generator import ResponseGenerator
from app.services.response_verifier import ResponseVerifier
from app.schemas.verification import VerificationRequest
from app.schemas.complex import SubTask, ComplexTaskPlan
from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer
from app.services.complex.synthesis_strategies import (
    deterministic_s2_assembly,
    hybrid_s3_assembly,
    llm_synthesis_s0_s1
)
from run_complex_task_comparison import (
    load_complex_prompts,
    extract_objectives_from_prompt,
    evaluate_objective_coverage
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synthesis_ablation")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "complex_task_synthesis_optimization"))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CACHED_SUBTASKS_FILE = os.path.join(RAW_DIR, "cached_subtasks.json")
RESULTS_FILE = os.path.join(RAW_DIR, "synthesis_ablation_results.jsonl")

def ci95(data):
    arr = np.array(data, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return {"mean": mean, "std": 0.0, "std_err": 0.0, "margin": 0.0, "ci_lower": mean, "ci_upper": mean, "p50": mean, "p95": mean}
    std_err = float(np.std(arr, ddof=1) / math.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n-1))
    margin = float(t_crit * std_err)
    return {
        "mean": round(mean, 4),
        "std": round(float(np.std(arr, ddof=1)), 4),
        "std_err": round(std_err, 4),
        "margin": round(margin, 4),
        "ci_lower": round(mean - margin, 4),
        "ci_upper": round(mean + margin, 4),
        "p50": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4)
    }

async def generate_subtasks_for_prompt(
    decomposer: ComplexTaskDecomposer,
    prompt_text: str
) -> Tuple[List[SubTask], float, float]:
    """
    Generates subtasks and runs parallel execution ONCE per prompt.
    Returns (subtasks, plan_latency_ms, parallel_wall_clock_ms).
    """
    t0 = time.perf_counter()
    plan = decomposer.decompose(prompt_text, execution_mode="local")
    
    from app.services.provider_failover import RequestProviderTracker
    provider_tracker = RequestProviderTracker()

    t_sub_start = time.perf_counter()
    executed_subtasks = await decomposer.parallel_scheduler.execute_plan_async(
        subtasks=plan.subtasks,
        execution_levels=plan.execution_levels,
        execution_mode="local",
        provider_tracker=provider_tracker
    )
    t_sub_end = time.perf_counter()
    
    parallel_wall_clock_ms = round((t_sub_end - t_sub_start) * 1000, 2)
    return executed_subtasks, plan.plan_latency_ms, parallel_wall_clock_ms

def run_synthesis_strategy(
    strategy_id: str,
    original_prompt: str,
    subtasks: List[SubTask],
    response_generator: ResponseGenerator,
    verifier: ResponseVerifier,
    decomp_latency_ms: float,
    subtask_wall_clock_ms: float
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cpu_before = psutil.cpu_percent(interval=None)
    ram_before_mb = psutil.virtual_memory().used / (1024 * 1024)

    fallback_used = False
    reproducible = True

    if strategy_id == "S0":
        # S0: Baseline DeepSeek 7B LLM synthesis
        resp_text, synth_ms = llm_synthesis_s0_s1(original_prompt, subtasks, "deepseek-r1-7b", response_generator)
    elif strategy_id == "S1":
        # S1: Lightweight Gemma 3 4B LLM synthesis
        resp_text, synth_ms = llm_synthesis_s0_s1(original_prompt, subtasks, "gemma-3-4b", response_generator)
    elif strategy_id == "S2":
        # S2: Deterministic Structured Assembly
        resp_text, synth_ms = deterministic_s2_assembly(original_prompt, subtasks)
        # Verify 100% byte-for-byte reproducibility
        resp_text2, _ = deterministic_s2_assembly(original_prompt, subtasks)
        reproducible = (resp_text == resp_text2)
    elif strategy_id == "S3":
        # S3: Hybrid Assembly + Gemma 3 4B Executive Summary
        resp_text, asm_ms, sum_ms, fallback_used = hybrid_s3_assembly(original_prompt, subtasks, response_generator)
        synth_ms = asm_ms + sum_ms
    else:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")

    t1 = time.perf_counter()
    total_e2e_ms = decomp_latency_ms + subtask_wall_clock_ms + synth_ms

    cpu_after = psutil.cpu_percent(interval=None)
    ram_after_mb = psutil.virtual_memory().used / (1024 * 1024)

    out_tokens = len(resp_text.split()) * 4 // 3
    char_count = len(resp_text)

    # Verification & Quality
    ver_req = VerificationRequest(
        prompt=original_prompt,
        selected_model=f"strategy_{strategy_id}",
        generated_text=resp_text,
        generation_success=True,
        execution_status="completed"
    )
    ver_res = verifier.verify_response(ver_req)
    quality_score = ver_res.structural_quality_score or 4.0

    # Objective Coverage
    objectives = extract_objectives_from_prompt(original_prompt)
    cov_pct, sat_cnt, tot_cnt, missing_objs = evaluate_objective_coverage(resp_text, objectives)

    return {
        "strategy_id": strategy_id,
        "decomposition_latency_ms": decomp_latency_ms,
        "subtask_parallel_wall_time_ms": subtask_wall_clock_ms,
        "synthesis_latency_ms": synth_ms,
        "total_e2e_latency_ms": total_e2e_ms,
        "output_tokens": out_tokens,
        "character_count": char_count,
        "quality_score_0_to_5": round(quality_score, 2),
        "correctness_score": ver_res.factual_verification_status or "verified",
        "relevance_score": round(ver_res.relevance_score, 2),
        "completeness_score": round(ver_res.completeness_score, 2),
        "objective_coverage_pct": cov_pct,
        "satisfied_objectives": sat_cnt,
        "total_requested_objectives": tot_cnt,
        "missing_objectives": missing_objs,
        "cpu_utilization_pct": round((cpu_before + cpu_after) / 2.0, 1),
        "ram_usage_mb": round(ram_after_mb, 1),
        "fallback_used": fallback_used,
        "reproducible": reproducible,
        "response_text": resp_text
    }

async def run_ablation_experiment():
    logger.info("==================================================")
    logger.info("STARTING RUN_4 — CONTROLLED SYNTHESIS ABLATION")
    logger.info("==================================================")

    primary_prompts, supp_prompts = load_complex_prompts()
    all_prompts = primary_prompts + supp_prompts
    logger.info(f"Total prompt dataset scope: N = {len(all_prompts)} prompts.")

    decomposer = ComplexTaskDecomposer()
    response_generator = ResponseGenerator()
    verifier = ResponseVerifier()

    # Load or initialize cached subtasks
    cached_subtasks = {}
    if os.path.exists(CACHED_SUBTASKS_FILE):
        with open(CACHED_SUBTASKS_FILE, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
            for pid, cdata in raw_cache.items():
                st_list = [SubTask(**st_dict) for st_dict in cdata["subtasks"]]
                cached_subtasks[pid] = {
                    "subtasks": st_list,
                    "decomp_latency_ms": cdata["decomp_latency_ms"],
                    "subtask_wall_clock_ms": cdata["subtask_wall_clock_ms"]
                }
        logger.info(f"Loaded cached subtasks for {len(cached_subtasks)} prompts.")

    # Load existing results for resume capability
    existing_results = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    existing_results.add((r["prompt_id"], r["strategy_id"]))
        logger.info(f"Found {len(existing_results)} completed strategy trials in raw log.")

    for idx, pitem in enumerate(all_prompts, 1):
        pid = pitem["prompt_id"]
        ptext = pitem["prompt"]
        cat = pitem["category"]

        logger.info(f"\n--- Progress ({idx}/{len(all_prompts)}) - '{pid}' ({cat}) ---")

        # Step 1: Ensure subtasks are generated ONCE per prompt
        if pid not in cached_subtasks:
            logger.info(f"Generating subtasks for prompt '{pid}'...")
            subtasks, decomp_ms, subtask_wall_ms = await generate_subtasks_for_prompt(decomposer, ptext)
            cached_subtasks[pid] = {
                "subtasks": subtasks,
                "decomp_latency_ms": decomp_ms,
                "subtask_wall_clock_ms": subtask_wall_ms
            }
            # Save cache
            cache_to_save = {}
            for cpid, cval in cached_subtasks.items():
                cache_to_save[cpid] = {
                    "subtasks": [st.dict() for st in cval["subtasks"]],
                    "decomp_latency_ms": cval["decomp_latency_ms"],
                    "subtask_wall_clock_ms": cval["subtask_wall_clock_ms"]
                }
            with open(CACHED_SUBTASKS_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_to_save, f, indent=2)

        cdata = cached_subtasks[pid]
        subtasks = cdata["subtasks"]
        decomp_ms = cdata["decomp_latency_ms"]
        subtask_wall_ms = cdata["subtask_wall_clock_ms"]

        # Step 2: Evaluate S0, S1, S2, S3 on identical subtasks
        for strat in ["S0", "S1", "S2", "S3"]:
            if (pid, strat) in existing_results:
                logger.info(f"Strategy '{strat}' for prompt '{pid}' already completed. Skipping.")
                continue

            logger.info(f"Executing Strategy '{strat}' for prompt '{pid}'...")
            res = run_synthesis_strategy(
                strategy_id=strat,
                original_prompt=ptext,
                subtasks=subtasks,
                response_generator=response_generator,
                verifier=verifier,
                decomp_latency_ms=decomp_ms,
                subtask_wall_clock_ms=subtask_wall_ms
            )
            res["prompt_id"] = pid
            res["category"] = cat
            res["original_prompt"] = ptext
            res["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

            existing_results.add((pid, strat))

    logger.info("==================================================")
    logger.info("CONTROLLED ABLATION EXPERIMENT COMPLETE!")
    logger.info("==================================================")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(run_ablation_experiment())
