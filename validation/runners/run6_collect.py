"""Controlled RUN 6 collection into an isolated research buffer.

This collector never writes the canonical production buffer or artifact. It uses
real orchestration responses and records failed provider executions as invalid
records through the normal Step 20 path.
"""
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.config import settings
from app.schemas.orchestration import OrchestrationRequest
from app.services.experience_buffer import ACTION_MAP, ExperienceBufferService
from app.services.orchestration_pipeline import OrchestrationPipeline

logging.getLogger("orchestrator").setLevel(logging.WARNING)

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "run6" / "final_attempt"
BUFFER_PATH = OUTPUT_DIR / "run6_experiences.jsonl"
CANONICAL_BUFFER_PATH = PROJECT_ROOT / "data" / "rl" / "experience_buffer.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "datasets" / "evaluation" / "benchmark_100.json"

TARGETS = [
    (0, "gemma-3-4b", "local"),
    (1, "qwen-coder-3b", "local"),
    (2, "deepseek-r1-7b", "local"),
    (3, "gemini-3.5-flash", "online"),
    (4, "mistral-small-latest", "online"),
    (5, "llama-3.3-70b-versatile", "online"),
    (6, "meta-llama/llama-3.3-70b-instruct", "online"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if BUFFER_PATH.exists():
        raise RuntimeError(f"Refusing to reuse existing RUN 6 output: {BUFFER_PATH}")
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(BENCHMARK_PATH)

    prompts = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    if len(prompts) < 8:
        raise RuntimeError("Benchmark prompt set is too small for controlled coverage collection.")

    canonical_before_count = sum(1 for line in CANONICAL_BUFFER_PATH.open(encoding="utf-8") if line.strip())
    canonical_before_hash = sha256_file(CANONICAL_BUFFER_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    settings.RL_DATA_COLLECTION_MODE = True
    settings.EXPLORATION_EPSILON = float(os.getenv("EXPLORATION_EPSILON", settings.EXPLORATION_EPSILON))
    settings.EXPLORATION_SEED = int(os.getenv("EXPLORATION_SEED", settings.EXPLORATION_SEED))

    buffer_service = ExperienceBufferService(capacity=1000, persistence_path=str(BUFFER_PATH))
    pipeline = OrchestrationPipeline(experience_buffer=buffer_service)
    all_models = list(ACTION_MAP.keys())
    attempts = []

    # One bounded real attempt per target avoids repeated calls to providers
    # already known to be quota-limited or unavailable.
    for round_index in range(1):
        for action_index, target_model, mode in TARGETS:
            prompt_item = prompts[(round_index * len(TARGETS) + action_index) % len(prompts)]
            mode_models = [model for _, model, target_mode in TARGETS if target_mode == mode]
            excluded = [model for model in mode_models if model != target_model]
            request = OrchestrationRequest(
                prompt=prompt_item["prompt"],
                execution_mode=mode,
                excluded_models=excluded,
            )
            started = time.time()
            try:
                response = pipeline.run_pipeline(request)
                record = buffer_service._buffer[-1]
                attempts.append({
                    "round": round_index + 1,
                    "target_action": action_index,
                    "target_model": target_model,
                    "mode": mode,
                    "success": bool(response.success and response.generation.execution_status == "completed"),
                    "selected_model": response.selected_model,
                    "executed_model": getattr(response.generation, "model_id", None),
                    "record_action": record.action,
                    "record_valid": record.is_valid_rl_sample,
                    "fallback_used": (record.metadata or {}).get("fallback_used"),
                    "fallback_reason": (record.metadata or {}).get("fallback_reason"),
                    "cost": (record.metadata or {}).get("cost"),
                    "cost_source": (record.metadata or {}).get("cost_source"),
                    "elapsed_seconds": round(time.time() - started, 3),
                })
                target_success = (
                    response.success
                    and response.generation.execution_status == "completed"
                    and record.executed_model == target_model
                    and not (record.metadata or {}).get("fallback_used", False)
                    and record.action == action_index
                )
                attempts[-1]["target_success"] = target_success
                print(f"[{len(attempts):02d}/7] target=A{action_index} {target_model} -> executed={record.executed_model} target_success={target_success}")
            except Exception as exc:
                attempts.append({
                    "round": round_index + 1,
                    "target_action": action_index,
                    "target_model": target_model,
                    "mode": mode,
                    "success": False,
                    "error": str(exc),
                })
                print(f"[{len(attempts):02d}/7] target=A{action_index} {target_model} -> ERROR {exc}")

    canonical_after_count = sum(1 for line in CANONICAL_BUFFER_PATH.open(encoding="utf-8") if line.strip())
    canonical_after_hash = sha256_file(CANONICAL_BUFFER_PATH)
    if canonical_after_count != canonical_before_count or canonical_after_hash != canonical_before_hash:
        raise RuntimeError(
            "FAIL SAFE: canonical experience buffer changed during RUN 6 collection; "
            "collection is invalid and no training is permitted."
        )

    raw_records = [json.loads(line) for line in BUFFER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid_records = [r for r in raw_records if len(r.get("state", [])) == 12 and isinstance(r.get("action"), int) and 0 <= r["action"] <= 6 and r.get("is_valid_rl_sample")]
    action_counts = Counter(r.get("action") for r in valid_records)
    manifest = {
        "run_id": "RUN_6_CONTROLLED_COLLECTION",
        "benchmark_path": str(BENCHMARK_PATH.relative_to(PROJECT_ROOT)),
        "benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "buffer_path": str(BUFFER_PATH.relative_to(PROJECT_ROOT)),
        "buffer_sha256": sha256_file(BUFFER_PATH),
        "attempt_count": len(attempts),
        "raw_record_count": len(raw_records),
        "valid_record_count": len(valid_records),
        "action_counts": {f"A{index}": action_counts.get(index, 0) for index in range(8)},
        "exploration_mode": settings.RL_DATA_COLLECTION_MODE,
        "exploration_epsilon": settings.EXPLORATION_EPSILON,
        "exploration_seed": settings.EXPLORATION_SEED,
        "a7_generation_count": action_counts.get(7, 0),
        "attempts": attempts,
        "timestamp": time.time(),
        "canonical_buffer_before": {"count": canonical_before_count, "sha256": canonical_before_hash},
        "canonical_buffer_after": {"count": canonical_after_count, "sha256": canonical_after_hash},
        "canonical_buffer_unchanged": True,
        "target_success_count": sum(1 for attempt in attempts if attempt.get("target_success")),
        "training_gate_passed": False,
    }
    (OUTPUT_DIR / "run6_collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("raw_record_count", "valid_record_count", "action_counts", "a7_generation_count", "buffer_sha256")}, indent=2))


if __name__ == "__main__":
    main()
