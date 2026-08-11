import time
import uuid
import logging
from typing import Optional, List, Dict, Any

from app.services.input_processor import InputProcessor
from app.services.intent_classifier import IntentClassifier
from app.services.complexity_analyzer import ComplexityAnalyzer
from app.services.resource_analyzer import ResourceAnalyzer
from app.services.model_registry import ModelRegistry
from app.services.policies.base_policy import BaseDecisionPolicy
from app.services.policies.baseline_policy import BaselineAdaptivePolicy
from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.services.providers.online_provider_manager import OnlineProviderManager
from app.services.experience_buffer import StateEncoder

from app.schemas.decision import (
    DecisionRequest,
    DecisionResponse,
    DecisionTrace,
    DecisionStatusResponse,
    CandidateScoreBreakdown
)

logger = logging.getLogger("orchestrator")

class AdaptiveDecisionEngine:
    def __init__(self, policy: Optional[BaseDecisionPolicy] = None):
        self.input_processor = InputProcessor()
        self.intent_classifier = IntentClassifier()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.resource_analyzer = ResourceAnalyzer()
        self.model_registry = ModelRegistry()
        self.online_provider_manager = OnlineProviderManager()
        self.state_encoder = StateEncoder()
        
        self.policy = policy or BaselineAdaptivePolicy()
        self.shadow_rl_policy = RLContextualBanditPolicy()

    def get_status(self) -> DecisionStatusResponse:
        models = self.model_registry.list_models()
        registered_count = len(models)
        executable_llm_models = [
            m.model_id for m in models
            if m.model_type == "llm" and m.available and m.configuration_status == "configured"
        ]

        engine_status = "ready" if executable_llm_models else "no_executable_models"

        return DecisionStatusResponse(
            status=engine_status,
            policy=self.policy.policy_name,
            registered_models_count=registered_count,
            executable_candidates_count=len(executable_llm_models),
            executable_models=executable_llm_models
        )

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        t_pipeline_start = time.perf_counter()

        if not request.text or not request.text.strip():
            raise ValueError("Prompt text for adaptive decision engine cannot be empty or contain only whitespace.")

        processed_input = self.input_processor.process_prompt(request.text)
        cleaned_prompt = processed_input.cleaned_prompt

        t_analysis_start = time.perf_counter()

        # Step 9 Intent Analysis
        if request.intent:
            intent_dict = request.intent.dict()
        else:
            intent_res = self.intent_classifier.classify_intent(cleaned_prompt)
            intent_dict = {
                "intent": intent_res.intent,
                "top_similarity": intent_res.top_similarity,
                "second_similarity": intent_res.second_similarity,
                "margin": intent_res.margin,
                "is_ambiguous": intent_res.is_ambiguous
            }

        # Step 10 Complexity Analysis
        if request.complexity:
            complexity_dict = request.complexity.dict()
        else:
            complexity_res = self.complexity_analyzer.analyze_complexity(cleaned_prompt)
            complexity_dict = {
                "level": complexity_res.complexity_level,
                "complexity_score": complexity_res.complexity_score,
                "semantic_complexity": complexity_res.factors.semantic_complexity,
                "reasoning_complexity": complexity_res.factors.reasoning_complexity,
                "task_complexity": complexity_res.factors.task_complexity,
                "context_complexity": complexity_res.factors.context_complexity,
                "output_complexity": complexity_res.factors.output_complexity
            }

        # Step 11 Resource Telemetry
        if request.resources:
            resource_dict = request.resources.dict()
        else:
            resource_res = self.resource_analyzer.get_resource_snapshot()
            resource_dict = {
                "cpu_utilization_percent": resource_res.cpu.utilization_percent,
                "memory_utilization_percent": resource_res.memory.utilization_percent,
                "gpu_available": resource_res.gpu.available,
                "available_memory_gb": resource_res.memory.available_gb,
                "execution_device": resource_res.runtime.execution_device,
                "gpu_total_vram_gb": resource_res.gpu.total_vram_gb,
                "gpu_free_vram_gb": resource_res.gpu.free_vram_gb,
                "gpu_used_vram_gb": resource_res.gpu.used_vram_gb
            }

        t_policy_start = time.perf_counter()

        # Discover Candidates based on Execution Mode (Local vs Online API)
        exec_mode = getattr(request, "execution_mode", "local") or "local"
        excluded = getattr(request, "excluded_models", []) or []

        if exec_mode in ["online", "mistral", "gemini"]:
            raw_candidates = self.online_provider_manager.get_online_model_candidates()
            candidate_models = [
                m for m in raw_candidates
                if m.available is True
                and m.configuration_status == "configured"
                and m.model_id not in excluded
            ]
        else:
            raw_candidates = self.model_registry.list_models()
            candidate_models = [
                m for m in raw_candidates
                if m.available is True
                and m.configuration_status == "configured"
                and m.execution_mode == "local"
                and m.model_id != "BAAI/bge-m3"
                and m.model_id not in excluded
            ]

        # Evaluate Candidates through Single Production Policy Authority (BaselineAdaptivePolicy)
        selected_model, winning_score, candidate_breakdowns, reasoning = self.policy.evaluate_candidates(
            prompt=cleaned_prompt,
            intent_info=intent_dict,
            complexity_info=complexity_dict,
            resource_info=resource_dict,
            candidate_models=candidate_models
        )

        # Step 21: Shadow Mode RL Evaluation (does NOT override baseline selected_model)
        shadow_rl_dict = None
        try:
            temp_dec_res = DecisionResponse(
                text=cleaned_prompt,
                selected_model=selected_model,
                decision_score=winning_score,
                policy=self.policy.policy_name,
                reasoning=reasoning,
                candidates=candidate_breakdowns,
                decision_trace=DecisionTrace(
                    decision_id="temp",
                    intent=intent_dict.get("intent", "general_qa"),
                    is_ambiguous=intent_dict.get("is_ambiguous", False),
                    complexity_level=complexity_dict.get("level", "medium"),
                    complexity_score=complexity_dict.get("complexity_score", 0.50),
                    resource_summary=resource_dict,
                    selected_model=selected_model,
                    decision_score=winning_score,
                    policy=self.policy.policy_name,
                    decision_latency_ms=0.0
                ),
                intent_info=intent_dict,
                complexity_info=complexity_dict,
                total_pipeline_latency_ms=0.0
            )

            state_vector = self.state_encoder.encode_state(
                type("TempOrchRes", (), {"decision": temp_dec_res, "decision_score": winning_score})()
            )

            shadow_decision = self.shadow_rl_policy.predict_shadow_decision(
                state_vector=state_vector,
                baseline_selected_model=selected_model,
                candidate_models=candidate_models
            )
            shadow_rl_dict = shadow_decision.dict()
        except Exception as e:
            logger.warning(f"Shadow RL evaluation warning: {str(e)}")

        t_end = time.perf_counter()

        policy_latency_ms = round((t_end - t_policy_start) * 1000, 2)
        total_latency_ms = round((t_end - t_pipeline_start) * 1000, 2)

        decision_id = f"dec-{uuid.uuid4().hex[:8]}"

        decision_trace = DecisionTrace(
            decision_id=decision_id,
            intent=intent_dict.get("intent", "general_qa"),
            is_ambiguous=intent_dict.get("is_ambiguous", False),
            complexity_level=complexity_dict.get("level", "medium"),
            complexity_score=complexity_dict.get("complexity_score", 0.50),
            resource_summary=resource_dict,
            selected_model=selected_model,
            decision_score=winning_score,
            policy=self.policy.policy_name,
            shadow_rl_decision=shadow_rl_dict,
            decision_latency_ms=policy_latency_ms
        )

        logger.info(f"AdaptiveDecisionEngine decision complete. Selected: '{selected_model}' (score={winning_score:.4f}, latency={total_latency_ms}ms).")

        return DecisionResponse(
            text=cleaned_prompt,
            selected_model=selected_model,
            decision_score=winning_score,
            policy=self.policy.policy_name,
            reasoning=reasoning,
            candidates=candidate_breakdowns,
            decision_trace=decision_trace,
            shadow_rl_decision=shadow_rl_dict,
            intent_info=intent_dict,
            complexity_info=complexity_dict,
            total_pipeline_latency_ms=total_latency_ms
        )
