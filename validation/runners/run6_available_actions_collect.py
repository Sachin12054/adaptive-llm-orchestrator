"""Collect RUN 6 data for the currently executable action subset only."""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import settings
from app.schemas.orchestration import OrchestrationRequest
from app.services.experience_buffer import ACTION_MAP, ExperienceBufferService
from app.services.orchestration_pipeline import OrchestrationPipeline

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "run6" / "final_available_actions"
BUFFER_PATH = OUTPUT_DIR / "run6_available_actions.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "datasets" / "evaluation" / "benchmark_100.json"
CANONICAL_PATH = PROJECT_ROOT / "data" / "rl" / "experience_buffer.jsonl"

TARGETS = [
    (0, "gemma-3-4b", "local", {"factual", "general_qa", "explanation"}),
    (1, "qwen-coder-3b", "local", {"coding"}),
    (2, "deepseek-r1-7b", "local", {"reasoning", "mathematics", "coding"}),
    (6, "meta-llama/llama-3.3-70b-instruct", "online", {"reasoning", "coding", "explanation"}),
]
REPETITIONS = 18


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to reuse existing output directory: {OUTPUT_DIR}")
    prompts = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    canonical_before = {"count": sum(1 for line in CANONICAL_PATH.open(encoding="utf-8") if line.strip()), "sha256": sha256(CANONICAL_PATH)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    settings.RL_DATA_COLLECTION_MODE = True
    buffer = ExperienceBufferService(capacity=1000, persistence_path=str(BUFFER_PATH))
    pipeline = OrchestrationPipeline(experience_buffer=buffer)
    attempts = []

    for action_index, target_model, mode, categories in TARGETS:
        candidates = [item for item in prompts if item.get("intent", item.get("category")) in categories]
        if not candidates:
            raise RuntimeError(f"No benchmark prompts for {target_model}")
        excluded = [model for index, model, _, _ in TARGETS if model != target_model]
        for repetition in range(REPETITIONS):
            item = candidates[repetition % len(candidates)]
            request = OrchestrationRequest(
                prompt=item["prompt"],
                execution_mode=mode,
                excluded_models=excluded,
            )
            started = time.time()
            try:
                response = pipeline.run_pipeline(request)
                records = list(buffer._buffer)
                record = records[-1]
                target_success = (
                    response.success
                    and response.generation.execution_status == "completed"
                    and record.executed_model == target_model
                    and record.executed_action == action_index
                    and not (record.metadata or {}).get("fallback_used", False)
                    and record.is_valid_rl_sample
                )
                attempts.append({
                    "target_action": action_index,
                    "target_model": target_model,
                    "prompt_id": item["id"],
                    "selected_action": record.rl_selected_action,
                    "executed_action": record.executed_action,
                    "selected_model": record.rl_selected_model,
                    "executed_model": record.executed_model,
                    "target_success": target_success,
                    "record_valid": record.is_valid_rl_sample,
                    "propensity": record.propensity_probability,
                    "reward": record.reward,
                    "latency_ms": (record.metadata or {}).get("pipeline_latency_ms"),
                    "provider": record.executed_provider,
                    "fallback_used": (record.metadata or {}).get("fallback_used"),
                    "fallback_reason": (record.metadata or {}).get("fallback_reason"),
                    "cost": (record.metadata or {}).get("cost"),
                    "cost_source": (record.metadata or {}).get("cost_source"),
                    "elapsed_seconds": round(time.time() - started, 3),
                })
                print(f"{len(attempts):03d}/72 target=A{action_index} executed={record.executed_model} valid={target_success}")
            except Exception as exc:
                attempts.append({"target_action": action_index, "target_model": target_model, "target_success": False, "error": str(exc)})
                print(f"{len(attempts):03d}/72 target=A{action_index} ERROR={exc}")

    canonical_after = {"count": sum(1 for line in CANONICAL_PATH.open(encoding="utf-8") if line.strip()), "sha256": sha256(CANONICAL_PATH)}
    if canonical_before != canonical_after:
        raise RuntimeError(f"FAIL SAFE: canonical buffer changed: {canonical_before} -> {canonical_after}")

    raw = [json.loads(line) for line in BUFFER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [r for r in raw if r.get("is_valid_rl_sample") and 0 <= r.get("action", -1) <= 6]
    counts = Counter(r.get("action") for r in valid)
    manifest = {
        "run_id": "RUN_6_AVAILABILITY_CONSTRAINED_COLLECTION",
        "full_configured_action_space": [f"A{i}" for i in range(7)],
        "final_executable_action_space": ["A0", "A1", "A2", "A6"],
        "excluded_actions": {
            "A3": "Gemini 429 RESOURCE_EXHAUSTED quota",
            "A4": "Mistral 429 rate limit",
            "A5": "Groq 403 Forbidden authorization failure",
            "A7": "Embedding action permanently masked",
        },
        "benchmark_path": str(BENCHMARK_PATH.relative_to(PROJECT_ROOT)),
        "dataset_path": str(BUFFER_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": sha256(BUFFER_PATH),
        "attempt_count": len(attempts),
        "raw_record_count": len(raw),
        "valid_record_count": len(valid),
        "target_success_count": sum(1 for item in attempts if item.get("target_success")),
        "action_counts": {f"A{i}": counts.get(i, 0) for i in range(8)},
        "a7_generation_count": counts.get(7, 0),
        "exploration_mode": True,
        "exploration_epsilon": settings.EXPLORATION_EPSILON,
        "exploration_seed": settings.EXPLORATION_SEED,
        "canonical_buffer_before": canonical_before,
        "canonical_buffer_after": canonical_after,
        "canonical_buffer_unchanged": True,
        "attempts": attempts,
        "timestamp": time.time(),
    }
    (OUTPUT_DIR / "collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("raw_record_count", "valid_record_count", "target_success_count", "action_counts", "dataset_sha256")}, indent=2))


if __name__ == "__main__":
    main()
