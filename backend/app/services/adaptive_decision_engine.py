import random
import time
import uuid
import logging
import threading
from typing import Optional, List, Dict, Any

from app.core.config import settings
from app.services.input_processor import InputProcessor
from app.services.intent_classifier import IntentClassifier
from app.services.complexity_analyzer import ComplexityAnalyzer
from app.services.resource_analyzer import ResourceAnalyzer
from app.services.model_registry import ModelRegistry
from app.services.policies.base_policy import BaseDecisionPolicy
from app.services.policies.baseline_policy import BaselineAdaptivePolicy
from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.services.providers.online_provider_manager import OnlineProviderManager
from app.services.experience_buffer import StateEncoder, ACTION_MAP

from app.schemas.decision import (
    DecisionRequest,
    DecisionResponse,
    DecisionTrace,
    DecisionStatusResponse,
    CandidateScoreBreakdown
)

logger = logging.getLogger("orchestrator")

class AdaptiveDecisionEngine:
    """
    Production Adaptive Decision Engine.
    Routes inference requests using trained RLContextualBanditPolicy as the primary production policy,
    with BaselineAdaptivePolicy as a safe, deterministic production fallback gate.
    """
    def __init__(self, policy: Optional[BaseDecisionPolicy] = None):
        self.input_processor = InputProcessor()
        self.intent_classifier = IntentClassifier()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.resource_analyzer = ResourceAnalyzer()
        self.model_registry = ModelRegistry()
        self.online_provider_manager = OnlineProviderManager()
        self.state_encoder = StateEncoder()
        self._rng = random.Random(getattr(settings, "EXPLORATION_SEED", 42))
        self._rng_lock = threading.Lock()
        
        self.baseline_policy = BaselineAdaptivePolicy()
        self.rl_policy = RLContextualBanditPolicy()

        if policy is not None:
            self.policy = policy
        elif getattr(settings, "PRODUCTION_POLICY", "rl").lower() == "baseline":
            self.policy = self.baseline_policy
        else:
            self.policy = self.rl_policy

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
        excluded_providers = getattr(request, "excluded_providers", []) or []

        if exec_mode in ["online", "mistral", "gemini"]:
            raw_candidates = self.online_provider_manager.get_online_model_candidates()
            candidate_models = []
            for m in raw_candidates:
                if not m.available or m.configuration_status != "configured":
                    logger.info(f"CANDIDATE UNAVAILABLE: model={m.model_id} | provider={m.provider} | reason=API key or server unconfigured")
                elif m.model_id in excluded:
                    logger.info(f"CANDIDATE UNAVAILABLE: model={m.model_id} | provider={m.provider} | reason=Excluded due to execution fallback")
                elif m.provider in excluded_providers or (m.provider and any(ex.lower() in m.provider.lower() for ex in excluded_providers)):
                    logger.info(f"CANDIDATE UNAVAILABLE: model={m.model_id} | provider={m.provider} | reason=Provider '{m.provider}' marked exhausted/cooling down")
                else:
                    candidate_models.append(m)
        else:
            raw_candidates = self.model_registry.list_models()
            candidate_models = []
            for m in raw_candidates:
                if m.execution_mode != "local" or m.model_id == "BAAI/bge-m3":
                    continue
                if not m.available or m.configuration_status != "configured":
                    logger.info(f"CANDIDATE UNAVAILABLE: model={m.model_id} | provider={m.provider} | reason=Local server unconfigured")
                elif m.model_id in excluded:
                    logger.info(f"CANDIDATE UNAVAILABLE: model={m.model_id} | provider={m.provider} | reason=Excluded due to execution fallback")
                elif m.provider in excluded_providers or (m.provider and any(ex.lower() in m.provider.lower() for ex in excluded_providers)):
                    logger.info(f"CANDIDATE UNAVAILABLE: model={m.model_id} | provider={m.provider} | reason=Provider '{m.provider}' marked exhausted/cooling down")
                else:
                    candidate_models.append(m)

        # Primary Production Policy Evaluation & Action Validation Gate
        primary_policy_name = self.policy.policy_name
        fallback_used = False
        fallback_reason = None
        rl_selected_model = None

        selected_model, winning_score, candidate_breakdowns, reasoning = self.policy.evaluate_candidates(
            prompt=cleaned_prompt,
            intent_info=intent_dict,
            complexity_info=complexity_dict,
            resource_info=resource_dict,
            candidate_models=candidate_models
        )

        if primary_policy_name == "rl_contextual_bandit_policy":
            rl_selected_model = selected_model

        # Validation Check: If RL selected model is None, embedding model, or unavailable -> Fallback to Baseline
        if selected_model is None or selected_model == "BAAI/bge-m3" or not any(c.model_id == selected_model and c.eligible for c in candidate_breakdowns):
            fallback_used = True
            if selected_model is None:
                fallback_reason = "RL Policy returned no valid candidate model."
            elif selected_model == "BAAI/bge-m3":
                fallback_reason = "RL Policy selected forbidden embedding model."
            else:
                fallback_reason = f"RL selected model '{selected_model}' is ineligible or unconfigured."

            logger.warning(f"RL PRODUCTION FALLBACK GATE TRIGGERED: {fallback_reason} Re-evaluating via BaselineAdaptivePolicy.")

            fb_model, fb_score, fb_breakdowns, fb_reasoning = self.baseline_policy.evaluate_candidates(
                prompt=cleaned_prompt,
                intent_info=intent_dict,
                complexity_info=complexity_dict,
                resource_info=resource_dict,
                candidate_models=candidate_models
            )
            selected_model = fb_model
            winning_score = fb_score
            candidate_breakdowns = fb_breakdowns
            reasoning = [f"[FALLBACK GATE] {fallback_reason}"] + fb_reasoning

        # Controlled Exploration Mode Check (RUN 6)
        is_data_collection_mode = getattr(settings, "RL_DATA_COLLECTION_MODE", False)
        epsilon = float(getattr(settings, "EXPLORATION_EPSILON", 0.20))

        eligible_candidate_models = [
            c.model_id for c in candidate_breakdowns
            if c.eligible and c.model_id != "BAAI/bge-m3"
        ]
        k_valid = len(eligible_candidate_models)

        candidate_probs: Dict[str, float] = {}
        propensity_prob: float = 1.0

        if is_data_collection_mode and k_valid > 0:
            greedy_model = selected_model
            probs_per_eligible = epsilon / float(k_valid)
            
            for m_id in ACTION_MAP.keys():
                if m_id == "BAAI/bge-m3" or m_id not in eligible_candidate_models:
                    candidate_probs[m_id] = 0.0
                elif m_id == greedy_model:
                    candidate_probs[m_id] = round((1.0 - epsilon) + probs_per_eligible, 5)
                else:
                    candidate_probs[m_id] = round(probs_per_eligible, 5)

            with self._rng_lock:
                explore = self._rng.random() < epsilon
                explored_model = self._rng.choice(eligible_candidate_models) if explore and len(eligible_candidate_models) > 1 else None
            if explored_model is not None:
                selected_model = explored_model
                reasoning.append(f"[CONTROLLED EXPLORATION MODE] Selected candidate '{explored_model}' via behavior policy (epsilon={epsilon:.2f}).")

            propensity_prob = candidate_probs.get(selected_model, 1.0)
            active_policy_used = f"{primary_policy_name}_exploration" if not fallback_used else "baseline_adaptive_policy_exploration"
        else:
            active_policy_used = "baseline_adaptive_policy" if fallback_used else primary_policy_name
            for m_id in ACTION_MAP.keys():
                candidate_probs[m_id] = 1.0 if m_id == selected_model else 0.0
            propensity_prob = 1.0 if selected_model in ACTION_MAP else 0.0

        t_end = time.perf_counter()

        policy_latency_ms = round((t_end - t_policy_start) * 1000, 2)
        total_latency_ms = round((t_end - t_pipeline_start) * 1000, 2)

        decision_id = f"dec-{uuid.uuid4().hex[:8]}"

        rl_telemetry = {
            "production_policy": primary_policy_name,
            "policy_version": getattr(self.rl_policy, "policy_version", "2.0.0"),
            "rl_selected_model": rl_selected_model,
            "executed_model": selected_model,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "production_override": True if not fallback_used and primary_policy_name == "rl_contextual_bandit_policy" else False
        }

        decision_trace = DecisionTrace(
            decision_id=decision_id,
            intent=intent_dict.get("intent", "general_qa"),
            is_ambiguous=intent_dict.get("is_ambiguous", False),
            complexity_level=complexity_dict.get("level", "medium"),
            complexity_score=complexity_dict.get("complexity_score", 0.50),
            resource_summary=resource_dict,
            selected_model=selected_model,
            decision_score=winning_score,
            policy=active_policy_used,
            shadow_rl_decision=rl_telemetry,
            candidate_action_probabilities=candidate_probs,
            propensity_probability=propensity_prob,
            decision_latency_ms=policy_latency_ms
        )

        logger.info(f"AdaptiveDecisionEngine decision complete. Policy: '{active_policy_used}' | Selected: '{selected_model}' (score={winning_score:.4f}, fallback={fallback_used}, latency={total_latency_ms}ms).")

        return DecisionResponse(
            text=cleaned_prompt,
            selected_model=selected_model,
            decision_score=winning_score,
            policy=active_policy_used,
            reasoning=reasoning,
            candidates=candidate_breakdowns,
            decision_trace=decision_trace,
            shadow_rl_decision=rl_telemetry,
            intent_info=intent_dict,
            complexity_info=complexity_dict,
            total_pipeline_latency_ms=total_latency_ms
        )
