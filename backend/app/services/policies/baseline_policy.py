import logging
from typing import List, Dict, Any, Tuple, Optional
from app.schemas.model import ModelMetadata
from app.schemas.decision import CandidateScoreBreakdown
from app.services.policies.base_policy import BaseDecisionPolicy

logger = logging.getLogger("orchestrator")

# Centralized estimated GPU VRAM requirements (in GB) for local models
ESTIMATED_MODEL_VRAM_GB: Dict[str, float] = {
    "gemma-3-4b": 3.0,
    "qwen-coder-3b": 2.5,
    "deepseek-r1-7b": 4.5,
}

class BaselineAdaptivePolicy(BaseDecisionPolicy):
    @property
    def policy_name(self) -> str:
        return "baseline_adaptive_policy"

    def _calculate_capability_score(
        self, intent: str, is_ambiguous: bool, capabilities: List[str]
    ) -> float:
        intent_capability_map = {
            "coding": "coding",
            "mathematics": "mathematics",
            "reasoning": "reasoning",
            "explanation": "explanation",
            "translation": "translation",
            "summarization": "summarization",
            "factual": "general_qa",
            "creative": "general_qa",
            "conversational": "conversational"
        }

        required_cap = intent_capability_map.get(intent, "general_qa")
        has_direct_cap = required_cap in capabilities
        has_general_cap = ("general_qa" in capabilities or "conversational" in capabilities)

        if is_ambiguous:
            if "reasoning" in capabilities and (has_direct_cap or has_general_cap):
                return 0.95
            elif has_direct_cap:
                return 0.90
            elif has_general_cap:
                return 0.80
            else:
                return 0.50

        if has_direct_cap:
            return 1.00
        elif has_general_cap:
            return 0.75
        else:
            return 0.40

    def _calculate_complexity_fit_score(
        self, complexity_score: float, model: ModelMetadata
    ) -> float:
        m_id = model.model_id.lower()
        is_reasoning_model = "deepseek" in m_id or "pro" in m_id or ("reasoning" in model.capabilities and "general_qa" not in model.capabilities)
        is_lightweight_model = m_id in ["gemma-3-4b", "qwen-coder-3b"] or "flash" in m_id or "fast" in model.display_name.lower()

        if complexity_score < 0.35:
            if is_lightweight_model:
                return 1.00
            elif is_reasoning_model:
                return 0.60
            else:
                return 0.75
        elif complexity_score < 0.65:
            if is_lightweight_model:
                return 0.95
            elif is_reasoning_model:
                return 0.80
            else:
                return 0.85
        elif complexity_score < 0.85:
            if is_reasoning_model:
                return 1.00
            else:
                return 0.75
        else:
            if is_reasoning_model:
                return 1.00
            else:
                return 0.65

    def _calculate_resource_fit_score(self, resource_info: Dict[str, Any], model: ModelMetadata) -> float:
        memory_percent = resource_info.get("memory_utilization_percent", 50.0)
        cpu_percent = resource_info.get("cpu_utilization_percent", 20.0)

        # 1. System Host Load Penalty
        system_fit = 1.0
        if memory_percent > 90.0 or cpu_percent > 90.0:
            system_fit = 0.60
        elif memory_percent > 80.0 or cpu_percent > 80.0:
            system_fit = 0.80

        # Online cloud API models do not consume local host GPU VRAM
        if getattr(model, "execution_mode", "local") == "online" or model.provider != "ollama":
            return round(system_fit, 4)

        # 2. GPU VRAM Fit Evaluation for local Ollama models
        gpu_available = resource_info.get("gpu_available", False)
        free_vram_gb = resource_info.get("gpu_free_vram_gb")

        if gpu_available and free_vram_gb is not None:
            est_vram = getattr(model, "estimated_vram_gb", None) or ESTIMATED_MODEL_VRAM_GB.get(model.model_id, 3.5)
            
            # Allow a 0.3 GB margin for VRAM framework overhead
            vram_deficit = est_vram - (free_vram_gb + 0.3)

            if vram_deficit <= 0:
                vram_fit = 1.00  # Fits comfortably in VRAM
            elif vram_deficit <= 0.6:
                vram_fit = 0.60  # Tight fit, partial CPU offloading
            else:
                vram_fit = 0.40  # Exceeds free VRAM; uses Ollama CPU offload execution mode

            return round(min(system_fit, vram_fit), 4)

        return round(system_fit, 4)

    def _calculate_context_fit_score(self, prompt: str, model: ModelMetadata) -> float:
        estimated_prompt_tokens = max(1, len(prompt.split()) * 2)
        model_context_limit = model.context_length or 100000

        if estimated_prompt_tokens <= model_context_limit:
            return 1.00
        else:
            return 0.00

    def evaluate_candidates(
        self,
        prompt: str,
        intent_info: Dict[str, Any],
        complexity_info: Dict[str, Any],
        resource_info: Dict[str, Any],
        candidate_models: List[ModelMetadata]
    ) -> Tuple[Optional[str], float, List[CandidateScoreBreakdown], List[str]]:
        
        intent = intent_info.get("intent", "general_qa")
        is_ambiguous = intent_info.get("is_ambiguous", False)
        complexity_level = complexity_info.get("level", "medium")
        complexity_score = complexity_info.get("complexity_score", 0.50)

        breakdowns: List[CandidateScoreBreakdown] = []
        eligible_candidates: List[Tuple[ModelMetadata, CandidateScoreBreakdown]] = []

        # Step 1: Filter and score all registered models
        for model in candidate_models:
            if model.model_type != "llm":
                breakdowns.append(CandidateScoreBreakdown(
                    model_id=model.model_id,
                    provider=model.provider,
                    display_name=model.display_name,
                    eligible=False,
                    ineligible_reason=f"Model type '{model.model_type}' is not an LLM.",
                    candidate_score=0.0
                ))
                continue

            if not model.available or model.configuration_status != "configured":
                breakdowns.append(CandidateScoreBreakdown(
                    model_id=model.model_id,
                    provider=model.provider,
                    display_name=model.display_name,
                    eligible=False,
                    ineligible_reason=f"Model is unconfigured or unavailable ({model.configuration_status}).",
                    candidate_score=0.0
                ))
                continue

            cap_score = round(self._calculate_capability_score(intent, is_ambiguous, model.capabilities), 4)
            cmplx_score = round(self._calculate_complexity_fit_score(complexity_score, model), 4)
            res_score = round(self._calculate_resource_fit_score(resource_info, model), 4)
            ctx_score = round(self._calculate_context_fit_score(prompt, model), 4)

            # Mark resource-infeasible models ineligible when severe VRAM deficit is detected
            if res_score == 0.0:
                est_vram = getattr(model, "estimated_vram_gb", None) or ESTIMATED_MODEL_VRAM_GB.get(model.model_id, 3.5)
                free_vram = resource_info.get("gpu_free_vram_gb", 0.0)
                inelig_reason = f"Insufficient GPU VRAM for estimated runtime requirement (requires ~{est_vram:.1f} GB, available free ~{free_vram:.1f} GB)."

                breakdown = CandidateScoreBreakdown(
                    model_id=model.model_id,
                    provider=model.provider,
                    display_name=model.display_name,
                    eligible=False,
                    ineligible_reason=inelig_reason,
                    capability_score=cap_score,
                    complexity_fit_score=cmplx_score,
                    resource_fit_score=0.0,
                    context_fit_score=ctx_score,
                    candidate_score=0.0
                )
                breakdowns.append(breakdown)
                continue

            weighted_score = round(
                0.35 * cap_score +
                0.35 * cmplx_score +
                0.15 * res_score +
                0.15 * ctx_score,
                4
            )

            breakdown = CandidateScoreBreakdown(
                model_id=model.model_id,
                provider=model.provider,
                display_name=model.display_name,
                eligible=True,
                ineligible_reason=None,
                capability_score=cap_score,
                complexity_fit_score=cmplx_score,
                resource_fit_score=res_score,
                context_fit_score=ctx_score,
                candidate_score=weighted_score
            )

            breakdowns.append(breakdown)
            eligible_candidates.append((model, breakdown))

        breakdowns.sort(key=lambda b: (b.eligible, b.candidate_score), reverse=True)

        if not eligible_candidates:
            reasoning = [
                "No executable LLM candidate is currently configured in environment.",
                "Verify local Ollama server is running and models are installed."
            ]
            return None, 0.0, breakdowns, reasoning

        # Select winning candidate
        winning_model, winning_breakdown = max(eligible_candidates, key=lambda cb: cb[1].candidate_score)
        selected_model_id = winning_model.model_id
        winning_score = winning_breakdown.candidate_score

        reasoning = [
            f"Evaluated intent '{intent}' (ambiguous={is_ambiguous}) and prompt complexity '{complexity_level.upper()}' (score={complexity_score:.4f}).",
            f"Candidate '{winning_model.model_id}' achieved top candidate score ({winning_score:.4f}) based on capabilities, complexity fit, and VRAM resources.",
            f"Capability match score: {winning_breakdown.capability_score:.2f} | Complexity fit score: {winning_breakdown.complexity_fit_score:.2f} | Resource fit score: {winning_breakdown.resource_fit_score:.2f}.",
            f"Selected model is fully configured and satisfied all resource and context constraints."
        ]

        if is_ambiguous:
            reasoning.append("Note: Intent classification was flagged as ambiguous; model selection prioritized versatile general reasoning capability.")

        return selected_model_id, winning_score, breakdowns, reasoning
