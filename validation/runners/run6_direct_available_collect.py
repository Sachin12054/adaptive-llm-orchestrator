"""Direct target execution collector for availability-constrained RUN 6.

Uses real ModelManager provider dispatch plus the production verifier/reward
components. It bypasses policy rerouting only to ensure the tested target model
is the model actually executed; failures are retained as invalid attempts.
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings
from app.schemas.decision import DecisionRequest
from app.schemas.provider import ProviderGenerationResponse
from app.schemas.response import ResponseGenerationResponse
from app.schemas.verification import VerificationRequest
from app.schemas.reward import RewardComputeRequest
from app.schemas.model_manager import ModelExecutionRequest
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.experience_buffer import ACTION_MAP, ExperienceBufferService
from app.services.model_manager import ModelManager
from app.services.response_verifier import ResponseVerifier
from app.services.reward_signal import RewardSignal

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "run6" / "final_available_actions_direct"
BUFFER_PATH = OUTPUT_DIR / "run6_available_actions.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "datasets" / "evaluation" / "benchmark_100.json"
CANONICAL_PATH = PROJECT_ROOT / "data" / "rl" / "experience_buffer.jsonl"
TARGETS = [(0, "gemma-3-4b", "local"), (1, "qwen-coder-3b", "local"), (2, "deepseek-r1-7b", "local"), (6, "meta-llama/llama-3.3-70b-instruct", "online")]
REPETITIONS = 18


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to reuse existing output: {OUTPUT_DIR}")
    prompts = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    canonical_before = {"count": sum(1 for line in CANONICAL_PATH.open(encoding="utf-8") if line.strip()), "sha256": sha256(CANONICAL_PATH)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    settings.RL_DATA_COLLECTION_MODE = True
    buffer = ExperienceBufferService(capacity=1000, persistence_path=str(BUFFER_PATH))
    manager = ModelManager()
    engine = AdaptiveDecisionEngine()
    verifier = ResponseVerifier()
    reward_signal = RewardSignal()
    attempts = []

    for action_index, target_model, mode in TARGETS:
        for repetition in range(REPETITIONS):
            item = prompts[(action_index * 17 + repetition) % len(prompts)]
            prompt = item["prompt"]
            started = time.time()
            decision = engine.decide(DecisionRequest(text=prompt, execution_mode=mode))
            execution = manager.execute(ModelExecutionRequest(model_id=target_model, prompt=prompt, execution_mode=mode, temperature=0.2, max_output_tokens=256))
            verification = verifier.verify_response(VerificationRequest(prompt=prompt, selected_model=target_model, generated_text=execution.generated_text, generation_success=execution.success, execution_status=execution.execution_status))
            reward = reward_signal.compute_reward(RewardComputeRequest(prompt=prompt, selected_model=target_model, winning_score=decision.decision_score, execution_success=execution.success, execution_status=execution.execution_status, generated_text=execution.generated_text, verification_status=verification.verification_status, verified=verification.verified, response_present=verification.response_present, structural_quality_score=verification.structural_quality_score, completeness_score=verification.completeness_score, relevance_score=verification.relevance_score, factual_verification_status=verification.factual_verification_status))
            state_source = SimpleNamespace(decision=decision, decision_score=decision.decision_score)
            state = buffer.state_encoder.encode_state(state_source)
            candidate_probs = {model: (1.0 if model == target_model else 0.0) for model in ACTION_MAP}
            target_success = execution.success and execution.execution_status == "completed" and execution.model_id == target_model
            record = None
            if target_success:
                from app.schemas.experience import ExperienceRecord
                record = ExperienceRecord(experience_id=f"exp-run6-{len(attempts):04d}", state=state, action=action_index, action_model_id=target_model, reward=reward.reward, next_state=None, done=True, timestamp=time.time(), rl_selected_action=action_index, rl_selected_model=target_model, executed_action=action_index, executed_model=target_model, executed_provider=execution.provider, is_valid_rl_sample=True, behavior_action=action_index, behavior_model_id=target_model, propensity_probability=1.0, candidate_action_probabilities=candidate_probs, propensity_available=True, metadata={"prompt_id": item["id"], "prompt": prompt[:100], "production_policy": "availability_constrained_target_execution", "provider": execution.provider, "execution_status": execution.execution_status, "verification_status": verification.verification_status, "reward_status": reward.reward_status, "pipeline_latency_ms": execution.latency_ms, "cost": execution.cost, "cost_currency": execution.cost_currency, "cost_source": execution.cost_source, "rl_selected_model": target_model, "rl_selected_action": action_index, "executed_model": target_model, "executed_action": action_index, "executed_provider": execution.provider, "fallback_used": False, "fallback_reason": None, "is_valid_rl_sample": True})
                buffer.append_experience(record)
            attempts.append({"target_action": action_index, "target_model": target_model, "prompt_id": item["id"], "selected_action": action_index, "executed_action": action_index if target_success else None, "executed_model": execution.model_id, "target_success": target_success, "valid": bool(record), "propensity": 1.0, "reward": reward.reward, "latency_ms": execution.latency_ms, "provider": execution.provider, "error": execution.error_message, "cost": execution.cost, "cost_source": execution.cost_source, "elapsed_seconds": round(time.time() - started, 3)})
            print(f"{len(attempts):03d}/72 A{action_index} target={target_model} executed={execution.model_id} success={target_success}")

    canonical_after = {"count": sum(1 for line in CANONICAL_PATH.open(encoding="utf-8") if line.strip()), "sha256": sha256(CANONICAL_PATH)}
    if canonical_before != canonical_after:
        raise RuntimeError(f"FAIL SAFE: canonical buffer changed: {canonical_before} -> {canonical_after}")
    raw = list(buffer._buffer)
    counts = Counter(r.action for r in raw)
    manifest = {"run_id": "RUN_6_AVAILABILITY_CONSTRAINED_DIRECT_COLLECTION", "full_configured_action_space": [f"A{i}" for i in range(7)], "final_executable_action_space": ["A0", "A1", "A2", "A6"], "excluded_actions": {"A3": "Gemini 429 RESOURCE_EXHAUSTED quota", "A4": "Mistral 429 rate limit", "A5": "Groq 403 Forbidden authorization failure", "A7": "Embedding action permanently masked"}, "dataset_path": str(BUFFER_PATH.relative_to(PROJECT_ROOT)), "dataset_sha256": sha256(BUFFER_PATH), "attempt_count": len(attempts), "raw_record_count": len(attempts), "valid_record_count": len(raw), "target_success_count": len(raw), "action_counts": {f"A{i}": counts.get(i, 0) for i in range(8)}, "a7_generation_count": counts.get(7, 0), "exploration_mode": True, "exploration_epsilon": settings.EXPLORATION_EPSILON, "exploration_seed": settings.EXPLORATION_SEED, "canonical_buffer_before": canonical_before, "canonical_buffer_after": canonical_after, "canonical_buffer_unchanged": True, "attempts": attempts, "timestamp": time.time()}
    (OUTPUT_DIR / "collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("raw_record_count", "valid_record_count", "target_success_count", "action_counts", "dataset_sha256")}, indent=2))

if __name__ == "__main__":
    main()
