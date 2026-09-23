import os
import json
import time
import uuid
import random
import logging
from typing import List, Dict, Any, Optional, Tuple

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

# Phase 6 & RUN 6 — Authoritative Unified 8-Model Action Space Mapping & Alias Resolution (K=8)
ACTION_MAP: Dict[str, int] = {
    "gemma-3-4b": 0,
    "qwen-coder-3b": 1,
    "deepseek-r1-7b": 2,
    "gemini-3.5-flash": 3,
    "mistral-small-latest": 4,
    "llama-3.3-70b-versatile": 5,
    "meta-llama/llama-3.3-70b-instruct": 6,
    "BAAI/bge-m3": 7,
}

REVERSE_ACTION_MAP: Dict[int, str] = {
    0: "gemma-3-4b",
    1: "qwen-coder-3b",
    2: "deepseek-r1-7b",
    3: "gemini-3.5-flash",
    4: "mistral-small-latest",
    5: "llama-3.3-70b-versatile",
    6: "meta-llama/llama-3.3-70b-instruct",
    7: "BAAI/bge-m3",
}

MODEL_ALIASES: Dict[str, str] = {
    "gemma-3-4b": "gemma-3-4b",
    "qwen-coder-3b": "qwen-coder-3b",
    "deepseek-r1-7b": "deepseek-r1-7b",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-2.0-flash": "gemini-3.5-flash",
    "gemini-2.5-flash": "gemini-3.5-flash",
    "gemini-1.5-flash": "gemini-3.5-flash",
    "mistral-small-latest": "mistral-small-latest",
    "mistral-small": "mistral-small-latest",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "groq-llama-3.3-70b": "llama-3.3-70b-versatile",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "openrouter-llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "BAAI/bge-m3": "BAAI/bge-m3",
}

def resolve_canonical_action(model_id: Optional[str], provider: Optional[str] = None) -> Tuple[int, str]:
    """
    Authoritative canonical action resolution function for K=8 action space.
    Returns (action_idx, canonical_model_id). If model cannot be mapped to a valid RL action (0-7),
    returns (-1, model_id).
    """
    if not model_id:
        return -1, "unknown"

    canonical = MODEL_ALIASES.get(model_id, model_id)
    if canonical in ACTION_MAP:
        return ACTION_MAP[canonical], canonical

    m_lower = model_id.lower()
    p_lower = (provider or "").lower()

    if "gemini" in m_lower or "gemini" in p_lower:
        return ACTION_MAP["gemini-3.5-flash"], "gemini-3.5-flash"
    elif "mistral" in m_lower or "mistral" in p_lower:
        return ACTION_MAP["mistral-small-latest"], "mistral-small-latest"
    elif "groq" in m_lower or ("llama" in m_lower and "versatile" in m_lower):
        return ACTION_MAP["llama-3.3-70b-versatile"], "llama-3.3-70b-versatile"
    elif "openrouter" in m_lower or "instruct" in m_lower:
        return ACTION_MAP["meta-llama/llama-3.3-70b-instruct"], "meta-llama/llama-3.3-70b-instruct"
    elif "gemma" in m_lower:
        return ACTION_MAP["gemma-3-4b"], "gemma-3-4b"
    elif "qwen" in m_lower:
        return ACTION_MAP["qwen-coder-3b"], "qwen-coder-3b"
    elif "deepseek" in m_lower:
        return ACTION_MAP["deepseek-r1-7b"], "deepseek-r1-7b"
    elif "bge" in m_lower:
        return ACTION_MAP["BAAI/bge-m3"], "BAAI/bge-m3"

    return -1, model_id

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

        res_summary = getattr(decision_trace, "resource_summary", {})
        cpu_raw = res_summary.get("cpu_utilization_percent") if res_summary else None
        mem_raw = res_summary.get("memory_utilization_percent") if res_summary else None
        
        cpu_util = float(cpu_raw if cpu_raw is not None else 20.0) / 100.0
        mem_util = float(mem_raw if mem_raw is not None else 50.0) / 100.0
        gpu_avail = 1.0 if (res_summary and res_summary.get("gpu_available")) else 0.0

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
        has_custom_path = (
            kwargs.get("persistence_path") is not None
            or len(args) > 1
            or (len(args) == 1 and isinstance(args[0], str))
        )

        if not has_custom_path:
            if cls._instance is None:
                cls._instance = super(ExperienceBufferService, cls).__new__(cls)
                cls._instance._initialized = False
                cls._instance._is_default = True
            return cls._instance

        return super(ExperienceBufferService, cls).__new__(cls)

    def __init__(self, capacity: int = 1000, persistence_path: Optional[str] = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        default_path = os.path.abspath(os.path.join(project_root, "data", "rl", "experience_buffer.jsonl"))
        test_path = os.getenv("ADAPTIVE_TEST_BUFFER_PATH")

        if persistence_path is None and getattr(self, "_is_default", False):
            if getattr(self, "_initialized", False):
                return

        if getattr(self, "_initialized", False):
            return

        self.capacity = capacity
        self.state_encoder = StateEncoder()

        if persistence_path is None:
            self._is_default = True
            self.persistence_path = test_path or default_path
        else:
            self._is_default = False
            target_path = os.path.abspath(os.path.join(project_root, persistence_path)) if not os.path.isabs(persistence_path) else persistence_path
            self.persistence_path = target_path

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
                f.write(record.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist experience record to disk: {str(e)}")

    def get_status(self) -> ExperienceBufferStatusResponse:
        return ExperienceBufferStatusResponse(
            status="ready",
            service="experience_buffer",
            capacity=self.capacity,
            current_size=len(self._buffer),
            state_dim=self.state_encoder.state_dim,
            action_dim=len(ACTION_MAP),
            persistence_path=self.persistence_path
        )

    def append_experience(self, record: ExperienceRecord, persist: bool = True):
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)

        self._buffer.append(record)

        if persist:
            self._persist_record(record)

    def record_from_orchestration(self, orchestration_res: OrchestrationResponse) -> ExperienceRecord:
        state = self.state_encoder.encode_state(orchestration_res)
        exact_reward = float(orchestration_res.reward.reward)
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        timestamp = time.time()

        gen = getattr(orchestration_res, "generation", None)
        plan = getattr(orchestration_res, "complex_plan", None)

        cost = getattr(gen, "cost", 0.0) if gen else 0.0
        cost_currency = getattr(gen, "cost_currency", "USD") if gen else "USD"
        cost_source = getattr(gen, "cost_source", "zero_local") if gen else "zero_local"
        provider = getattr(gen, "provider", "ollama") if gen else "ollama"

        usage = getattr(gen, "usage", None) if gen else None
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        tot_tok = getattr(usage, "total_tokens", 0) if usage else 0

        if plan and getattr(plan, "total_workflow_cost", None) is not None:
            cost = plan.total_workflow_cost
            cost_source = getattr(plan, "cost_source", "workflow_aggregated") or "workflow_aggregated"

        shadow_info = getattr(orchestration_res.decision, "shadow_rl_decision", {}) or {}

        # RUN 6 Fix: Distinguish Selected vs Executed Action
        rl_selected_model = shadow_info.get("rl_selected_model") or orchestration_res.selected_model
        rl_selected_action, _ = resolve_canonical_action(rl_selected_model)

        executed_model = (gen.model_id if (gen and gen.model_id) else orchestration_res.selected_model) or "none"
        executed_action, canonical_executed_model = resolve_canonical_action(executed_model, provider)

        is_valid_rl_sample = (executed_action >= 0 and len(state) == 12)
        if executed_action == -1:
            logger.warning(
                f"Executed model '{executed_model}' (provider: '{provider}') could not be mapped to canonical ACTION_MAP. "
                f"Experience recorded with is_valid_rl_sample=False."
            )

        fallback_used = shadow_info.get("fallback_used", False)
        fallback_reason = shadow_info.get("fallback_reason")

        metadata = {
            "prompt": orchestration_res.prompt[:100],
            "action_model_id": canonical_executed_model,
            "provider": provider,
            "execution_status": gen.execution_status if gen else "completed",
            "verification_status": orchestration_res.verification.verification_status,
            "reward_status": orchestration_res.reward.reward_status,
            "pipeline_latency_ms": orchestration_res.pipeline_latency_ms,
            "cost": float(cost),
            "cost_currency": cost_currency,
            "cost_source": cost_source,
            "input_tokens": in_tok or 0,
            "output_tokens": out_tok or 0,
            "total_tokens": tot_tok or 0,
            "production_policy": getattr(orchestration_res.decision, "policy", "rl_contextual_bandit_policy"),
            "rl_selected_model": rl_selected_model,
            "rl_selected_action": rl_selected_action if rl_selected_action >= 0 else None,
            "executed_model": canonical_executed_model,
            "executed_action": executed_action if executed_action >= 0 else None,
            "executed_provider": provider,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "is_valid_rl_sample": is_valid_rl_sample
        }

        # Propensity Logging: Use decision trace logged probabilities if available
        dec_trace = getattr(orchestration_res.decision, "decision_trace", None)
        logged_probs = getattr(dec_trace, "candidate_action_probabilities", None) if dec_trace else None
        logged_prop = getattr(dec_trace, "propensity_probability", None) if dec_trace else None

        if logged_probs:
            candidate_probs = logged_probs
            propensity_prob = logged_prop if logged_prop is not None else candidate_probs.get(canonical_executed_model, 0.0)
        else:
            candidate_probs = {m: (1.0 if m == canonical_executed_model else 0.0) for m in ACTION_MAP.keys()}
            propensity_prob = 1.0 if executed_action >= 0 else 0.0

        record = ExperienceRecord(
            experience_id=exp_id,
            state=state,
            action=executed_action,
            action_model_id=canonical_executed_model,
            reward=exact_reward,
            next_state=None,
            done=True,
            timestamp=timestamp,
            rl_selected_action=rl_selected_action if rl_selected_action >= 0 else None,
            rl_selected_model=rl_selected_model,
            executed_action=executed_action if executed_action >= 0 else None,
            executed_model=canonical_executed_model,
            executed_provider=provider,
            is_valid_rl_sample=is_valid_rl_sample,
            behavior_action=executed_action,
            behavior_model_id=canonical_executed_model,
            propensity_probability=propensity_prob,
            candidate_action_probabilities=candidate_probs,
            propensity_available=True if propensity_prob > 0.0 else False,
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

    def get_cost_metrics(self) -> Dict[str, Any]:
        """
        Computes authoritative backend API cost metrics across recorded experiences.
        Distinguishes local zero-cost Ollama calls from online cloud API costs.
        """
        total_api_cost = 0.0
        cost_today = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        cloud_requests_count = 0
        local_requests_count = 0

        now = time.time()
        seconds_in_day = 86400

        cost_by_provider: Dict[str, Dict[str, Any]] = {}
        cost_by_model: Dict[str, Dict[str, Any]] = {}
        recent_history: List[Dict[str, Any]] = []

        for record in self._buffer:
            meta = record.metadata or {}
            cost = float(meta.get("cost", 0.0) or 0.0)
            provider = str(meta.get("provider", "ollama") or "ollama")
            model_id = str(record.action_model_id or meta.get("action_model_id", "unknown"))
            in_tok = int(meta.get("input_tokens", 0) or 0)
            out_tok = int(meta.get("output_tokens", 0) or 0)
            tot_tok = int(meta.get("total_tokens", 0) or (in_tok + out_tok))
            cost_source = str(meta.get("cost_source", "zero_local"))

            is_cloud = (cost_source != "zero_local" and "ollama" not in provider.lower())
            if is_cloud:
                cloud_requests_count += 1
            else:
                local_requests_count += 1

            total_api_cost += cost
            if (now - record.timestamp) <= seconds_in_day:
                cost_today += cost

            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_tokens += tot_tok

            # Provider Breakdown
            p_key = provider
            if p_key not in cost_by_provider:
                cost_by_provider[p_key] = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
            cost_by_provider[p_key]["requests"] += 1
            cost_by_provider[p_key]["input_tokens"] += in_tok
            cost_by_provider[p_key]["output_tokens"] += out_tok
            cost_by_provider[p_key]["total_cost"] = round(cost_by_provider[p_key]["total_cost"] + cost, 6)

            # Model Breakdown
            m_key = model_id
            if m_key not in cost_by_model:
                cost_by_model[m_key] = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}
            cost_by_model[m_key]["requests"] += 1
            cost_by_model[m_key]["input_tokens"] += in_tok
            cost_by_model[m_key]["output_tokens"] += out_tok
            cost_by_model[m_key]["total_cost"] = round(cost_by_model[m_key]["total_cost"] + cost, 6)

            recent_history.append({
                "experience_id": record.experience_id,
                "timestamp": record.timestamp,
                "model_id": model_id,
                "provider": provider,
                "cost": round(cost, 6),
                "cost_source": cost_source,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": tot_tok,
                "reward": record.reward
            })

        avg_cloud_cost = round(total_api_cost / cloud_requests_count, 6) if cloud_requests_count > 0 else 0.0

        return {
            "total_api_cost": round(total_api_cost, 6),
            "cost_today": round(cost_today, 6),
            "cost_session": round(total_api_cost, 6),
            "currency": "USD",
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "cloud_requests_count": cloud_requests_count,
            "local_requests_count": local_requests_count,
            "avg_cost_per_cloud_request": avg_cloud_cost,
            "cost_by_provider": cost_by_provider,
            "cost_by_model": cost_by_model,
            "recent_cost_history": recent_history[-20:]
        }

    def clear(self):
        self._buffer.clear()
        if os.path.exists(self.persistence_path):
            try:
                os.remove(self.persistence_path)
            except Exception:
                pass
