import os
import json
import time
import uuid
import random
import logging
from typing import List, Dict, Any, Optional

from app.schemas.orchestration import OrchestrationResponse
from app.schemas.experience import (
    ExperienceRecord,
    ExperienceBufferStatusResponse
)

logger = logging.getLogger("orchestrator")

INTENT_MAP = {
    "general_qa": 0,
    "factual": 1,
    "coding": 2,
    "mathematics": 3,
    "reasoning": 4,
    "summarization": 5,
    "explanation": 6,
    "creative": 7,
    "translation": 8,
    "conversational": 9
}

# Standardized LLM Action index mapping (Ollama-only)
ACTION_MAP = {
    "gemma-3-4b": 0,
    "qwen-coder-3b": 1,
    "deepseek-r1-7b": 2,
}

class StateEncoder:
    def __init__(self):
        self.state_dim = 12

    def encode_state(self, orchestration_res: OrchestrationResponse) -> List[float]:
        decision_trace = orchestration_res.decision.decision_trace

        intent_str = getattr(decision_trace, "intent", "general_qa").lower()
        intent_code = INTENT_MAP.get(intent_str, 0) / 9.0

        is_ambiguous = 1.0 if getattr(decision_trace, "is_ambiguous", False) else 0.0
        complexity_score = float(getattr(decision_trace, "complexity_score", 0.50))
        decision_score = float(orchestration_res.decision_score)

        # Retrieve structural factors if present in decision object
        comp_input = getattr(orchestration_res.decision, "complexity", None)
        if comp_input:
            sem_comp = float(getattr(comp_input, "semantic_complexity", complexity_score))
            reas_comp = float(getattr(comp_input, "reasoning_complexity", complexity_score))
            task_comp = float(getattr(comp_input, "task_complexity", complexity_score))
            ctx_comp = float(getattr(comp_input, "context_complexity", complexity_score))
            out_comp = float(getattr(comp_input, "output_complexity", complexity_score))
        else:
            sem_comp = complexity_score
            reas_comp = complexity_score
            task_comp = complexity_score
            ctx_comp = complexity_score
            out_comp = complexity_score

        # Retrieve system telemetry
        res_summary = getattr(decision_trace, "resource_summary", {})
        cpu_util = float(res_summary.get("cpu_utilization_percent", 20.0)) / 100.0
        mem_util = float(res_summary.get("memory_utilization_percent", 50.0)) / 100.0
        gpu_avail = 1.0 if res_summary.get("gpu_available", False) else 0.0

        state_vector = [
            round(intent_code, 4),
            round(is_ambiguous, 4),
            round(complexity_score, 4),
            round(sem_comp, 4),
            round(reas_comp, 4),
            round(task_comp, 4),
            round(ctx_comp, 4),
            round(out_comp, 4),
            round(cpu_util, 4),
            round(mem_util, 4),
            round(gpu_avail, 4),
            round(decision_score, 4)
        ]

        return state_vector

class ExperienceBufferService:
    _instance: Optional["ExperienceBufferService"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExperienceBufferService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, capacity: int = 1000, persistence_path: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self.capacity = capacity
        self.state_encoder = StateEncoder()
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        raw_path = persistence_path or os.path.join("data", "rl", "experience_buffer.jsonl")
        self.persistence_path = raw_path if os.path.isabs(raw_path) else os.path.join(project_root, raw_path)
        
        self._buffer: List[ExperienceRecord] = []
        self._load_from_persistence()
        self._initialized = True

    def _load_from_persistence(self):
        if not os.path.exists(self.persistence_path):
            return

        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record_dict = json.loads(line)
                        record = ExperienceRecord(**record_dict)
                        self.append_experience(record, persist=False)
            logger.info(f"Loaded {len(self._buffer)} experience records into buffer from '{self.persistence_path}'")
        except Exception as e:
            logger.warning(f"Could not load existing experience buffer file: {str(e)}")

    def _persist_record(self, record: ExperienceRecord):
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            with open(self.persistence_path, "a", encoding="utf-8") as f:
                f.write(record.json() + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist experience record to disk: {str(e)}")

    def get_status(self) -> ExperienceBufferStatusResponse:
        return ExperienceBufferStatusResponse(
            status="ready",
            service="experience_buffer",
            capacity=self.capacity,
            current_size=len(self._buffer),
            state_dim=self.state_encoder.state_dim,
            persistence_path=self.persistence_path
        )

    def append_experience(self, record: ExperienceRecord, persist: bool = True):
        # FIFO Eviction when buffer is full
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)

        self._buffer.append(record)

        if persist:
            self._persist_record(record)

    def record_from_orchestration(self, orchestration_res: OrchestrationResponse) -> ExperienceRecord:
        # Encode state vector
        state = self.state_encoder.encode_state(orchestration_res)

        # Map discrete action index
        model_id = orchestration_res.selected_model or "none"
        action_idx = ACTION_MAP.get(model_id, 0 if model_id != "none" else -1)

        # Consume Step 18 reward cleanly
        exact_reward = float(orchestration_res.reward.reward)

        # Construct ExperienceRecord
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        timestamp = time.time()

        metadata = {
            "prompt": orchestration_res.prompt[:100],  # Short prompt snippet for auditability
            "action_model_id": model_id,
            "execution_status": orchestration_res.generation.execution_status,
            "verification_status": orchestration_res.verification.verification_status,
            "reward_status": orchestration_res.reward.reward_status,
            "pipeline_latency_ms": orchestration_res.pipeline_latency_ms
        }

        record = ExperienceRecord(
            experience_id=exp_id,
            state=state,
            action=action_idx,
            action_model_id=model_id,
            reward=exact_reward,
            next_state=None,  # Explicitly None for terminal single-turn requests
            done=True,        # Explicitly True for terminal single-turn requests
            timestamp=timestamp,
            metadata=metadata
        )

        self.append_experience(record, persist=True)
        return record

    def sample_batch(self, batch_size: int) -> List[ExperienceRecord]:
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from an empty experience buffer.")

        if batch_size <= 0 or batch_size > len(self._buffer):
            raise ValueError(f"Requested batch_size ({batch_size}) is invalid for buffer size ({len(self._buffer)}).")

        return random.sample(self._buffer, batch_size)

    def clear(self):
        self._buffer.clear()
        if os.path.exists(self.persistence_path):
            try:
                os.remove(self.persistence_path)
            except Exception:
                pass
