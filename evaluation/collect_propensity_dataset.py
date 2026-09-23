import os
import sys
import json
import time
import uuid
import numpy as np
import logging
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.response import ResponseGenerationRequest
from app.schemas.verification import VerificationRequest
from app.schemas.reward import RewardComputeRequest
from app.schemas.experience import ExperienceRecord
from app.services.experience_buffer import ExperienceBufferService, ACTION_MAP, REVERSE_ACTION_MAP
from app.services.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("propensity_collector")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bench_prompts_path = os.path.join(project_root, "datasets", "evaluation", "benchmark_100.json")
    out_jsonl_path = os.path.join(project_root, "data", "evaluation", "propensity_benchmark_100.jsonl")
    
    if not os.path.exists(bench_prompts_path):
        logger.error(f"Benchmark prompts file not found at '{bench_prompts_path}'")
        return

    with open(bench_prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    logger.info(f"Loaded {len(prompts)} prompts for controlled propensity collection.")

    pipeline = OrchestrationPipeline()
    model_registry = ModelRegistry()
    exp_service = ExperienceBufferService()
    
    de_engine = pipeline.decision_engine

    epsilon = 0.20
    existing_ids = set()

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    if os.path.exists(out_jsonl_path):
        with open(out_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        existing_ids.add(item["prompt_id"])
                    except Exception:
                        pass

    logger.info(f"Found {len(existing_ids)} previously collected propensity records. Resuming collection...")

    for idx, p_item in enumerate(prompts, start=1):
        prompt_id = p_item["id"]
        if prompt_id in existing_ids:
            logger.info(f"[{idx}/{len(prompts)}] Skipping already processed prompt '{prompt_id}'.")
            continue

        prompt_text = p_item["prompt"]
        exp_intent = p_item.get("intent") or p_item.get("category", "general_qa")
        exp_complexity = p_item["expected_complexity"]

        logger.info(f"[{idx}/{len(prompts)}] Collecting propensity data for '{prompt_id}' ({exp_intent} | {exp_complexity})...")

        # Step 1: Execute intent and complexity classification via Decision Engine components
        intent_res = de_engine.intent_classifier.classify_intent(prompt_text)
        complexity_res = de_engine.complexity_analyzer.analyze_complexity(prompt_text)
        resource_snapshot = de_engine.resource_analyzer.get_resource_snapshot()

        intent_info = {
            "intent": intent_res.intent,
            "is_ambiguous": intent_res.is_ambiguous
        }
        
        complexity_info = {
            "complexity_score": complexity_res.complexity_score,
            "complexity_level": complexity_res.complexity_level,
            "semantic_complexity": complexity_res.factors.semantic_complexity,
            "reasoning_complexity": complexity_res.factors.reasoning_complexity,
            "task_complexity": complexity_res.factors.task_complexity,
            "context_complexity": complexity_res.factors.context_complexity,
            "output_complexity": complexity_res.factors.output_complexity
        }

        telemetry_info = {
            "cpu_utilization_percent": resource_snapshot.cpu.utilization_percent,
            "memory_utilization_percent": resource_snapshot.memory.utilization_percent,
            "gpu_available": resource_snapshot.gpu.available
        }

        # Filter candidate LLM models
        all_models = model_registry.list_models()
        candidate_llms = [m for m in all_models if m.model_type == "llm" and m.available and m.configuration_status == "configured"]
        if not candidate_llms:
            candidate_llms = [m for m in all_models if m.model_type == "llm"]

        k_cand = len(candidate_llms)

        # Baseline preferred decision
        pref_model_id, pref_score, breakdowns, reasoning = de_engine.policy.evaluate_candidates(
            prompt_text, intent_info, complexity_info, telemetry_info, candidate_llms
        )

        if not pref_model_id:
            pref_model_id = candidate_llms[0].model_id

        pref_action_idx = ACTION_MAP.get(pref_model_id, 0)

        # Controlled Epsilon-Greedy Exploration Selection
        rng = np.random.default_rng(seed=42 + idx)
        if rng.uniform(0.0, 1.0) < (1.0 - epsilon):
            selected_model_id = pref_model_id
            selected_action_idx = pref_action_idx
            policy_type = "baseline_preferred"
        else:
            rand_cand = rng.choice(candidate_llms)
            selected_model_id = rand_cand.model_id
            selected_action_idx = ACTION_MAP.get(selected_model_id, 0)
            policy_type = "epsilon_explored"

        # Calculate exact propensity distribution P(a|s)
        p_other = epsilon / float(k_cand)
        p_preferred = (1.0 - epsilon) + p_other

        candidate_probs = {}
        for m in candidate_llms:
            if m.model_id == pref_model_id:
                candidate_probs[m.model_id] = round(p_preferred, 5)
            else:
                candidate_probs[m.model_id] = round(p_other, 5)

        selected_propensity = candidate_probs.get(selected_model_id, p_other)

        # Dispatch real model execution via ResponseGenerator
        cand_meta = next((m for m in candidate_llms if m.model_id == selected_model_id), candidate_llms[0])
        gen_req = ResponseGenerationRequest(
            prompt=prompt_text,
            selected_model=selected_model_id
        )

        gen_res = pipeline.response_generator.generate_response(gen_req)

        # Response verification and reward computation
        ver_req = VerificationRequest(
            prompt=prompt_text,
            selected_model=selected_model_id,
            generated_text=gen_res.generated_text,
            generation_success=gen_res.success,
            execution_status=gen_res.execution_status
        )
        ver_res = pipeline.response_verifier.verify_response(ver_req)

        reward_req = RewardComputeRequest(
            prompt=prompt_text,
            selected_model=selected_model_id,
            winning_score=pref_score,
            execution_success=gen_res.success,
            execution_status=gen_res.execution_status,
            generated_text=gen_res.generated_text,
            verification_status=ver_res.verification_status,
            verified=ver_res.verified,
            response_present=ver_res.response_present,
            structural_quality_score=ver_res.structural_quality_score,
            completeness_score=ver_res.completeness_score,
            relevance_score=ver_res.relevance_score,
            factual_verification_status=ver_res.factual_verification_status
        )
        reward_res = pipeline.reward_signal.compute_reward(reward_req)

        # Encode state vector
        dummy_res_obj = type("OrchestrationResponseMock", (), {
            "prompt": prompt_text,
            "decision_score": pref_score,
            "decision": type("DecisionMock", (), {
                "decision_trace": type("TraceMock", (), {
                    "intent": intent_info.get("intent"),
                    "is_ambiguous": intent_info.get("is_ambiguous"),
                    "complexity_score": complexity_info.get("complexity_score"),
                    "resource_summary": telemetry_info
                })(),
                "complexity": type("CmplxMock", (), complexity_info)()
            })()
        })()

        state_vector = exp_service.state_encoder.encode_state(dummy_res_obj)

        record_item = {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "expected_intent": exp_intent,
            "detected_intent": intent_info.get("intent"),
            "expected_complexity": exp_complexity,
            "detected_complexity": complexity_info.get("complexity_level"),
            "complexity_score": complexity_info.get("complexity_score"),
            "state_vector": state_vector,
            "baseline_preferred_model": pref_model_id,
            "behavior_action": selected_action_idx,
            "behavior_model_id": selected_model_id,
            "provider": cand_meta.provider,
            "propensity_probability": selected_propensity,
            "candidate_action_probabilities": candidate_probs,
            "policy_type": policy_type,
            "exploration_epsilon": epsilon,
            "propensity_available": True,
            "reward": float(reward_res.reward),
            "execution_success": gen_res.success,
            "execution_status": gen_res.execution_status,
            "error_category": getattr(gen_res, "error_category", "none") if hasattr(gen_res, "error_category") else ("none" if gen_res.success else "provider_failure"),
            "execution_latency_ms": gen_res.latency_ms,
            "input_tokens": gen_res.usage.input_tokens if gen_res.usage else 0,
            "output_tokens": gen_res.usage.output_tokens if gen_res.usage else 0,
            "timestamp": time.time()
        }

        existing_ids.add(prompt_id)

        # Write to machine-readable JSONL
        with open(out_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_item) + "\n")

        # Also persist to experience_buffer
        exp_record_obj = ExperienceRecord(
            experience_id=f"exp-prop-{uuid.uuid4().hex[:8]}",
            state=state_vector,
            action=selected_action_idx,
            action_model_id=selected_model_id,
            reward=float(reward_res.reward),
            next_state=None,
            done=True,
            timestamp=time.time(),
            behavior_action=selected_action_idx,
            behavior_model_id=selected_model_id,
            propensity_probability=selected_propensity,
            candidate_action_probabilities=candidate_probs,
            propensity_available=True,
            metadata={
                "prompt_id": prompt_id,
                "prompt": prompt_text[:100],
                "policy_type": policy_type,
                "exploration_epsilon": epsilon,
                "execution_status": gen_res.execution_status
            }
        )
        exp_service.append_experience(exp_record_obj, persist=True)

    print(f"\nSuccessfully completed propensity collection. Total stored: {len(existing_ids)} records.")
    print(f"Saved propensity benchmark dataset to '{out_jsonl_path}'")

if __name__ == "__main__":
    main()
