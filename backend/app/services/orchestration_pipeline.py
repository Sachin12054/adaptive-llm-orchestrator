import time
import logging
from typing import Optional

from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.response_generator import ResponseGenerator
from app.services.response_verifier import ResponseVerifier
from app.services.reward_signal import RewardSignal
from app.services.experience_buffer import ExperienceBufferService

from app.schemas.decision import DecisionRequest
from app.schemas.response import ResponseGenerationRequest, ResponseGenerationResponse
from app.schemas.verification import VerificationRequest
from app.schemas.reward import RewardComputeRequest
from app.schemas.orchestration import (
    OrchestrationRequest,
    OrchestrationResponse,
    OrchestrationStatusResponse
)

logger = logging.getLogger("orchestrator")

class OrchestrationPipeline:
    def __init__(
        self,
        decision_engine: Optional[AdaptiveDecisionEngine] = None,
        response_generator: Optional[ResponseGenerator] = None,
        response_verifier: Optional[ResponseVerifier] = None,
        reward_signal: Optional[RewardSignal] = None,
        experience_buffer: Optional[ExperienceBufferService] = None
    ):
        self.decision_engine = decision_engine or AdaptiveDecisionEngine()
        self.response_generator = response_generator or ResponseGenerator()
        self.response_verifier = response_verifier or ResponseVerifier()
        self.reward_signal = reward_signal or RewardSignal()
        self.experience_buffer = experience_buffer or ExperienceBufferService()

    def get_status(self) -> OrchestrationStatusResponse:
        de_status = self.decision_engine.get_status()
        rg_status = self.response_generator.get_status()
        
        is_ready = (rg_status.status == "ready")

        return OrchestrationStatusResponse(
            status="ready" if is_ready else "unconfigured",
            service="e2e_orchestrator",
            gemini_configured=is_ready,
            active_policy=de_status.policy
        )

    def run_pipeline(self, request: OrchestrationRequest) -> OrchestrationResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            raise ValueError("Prompt text for E2E orchestration pipeline cannot be empty or contain only whitespace.")

        logger.info("Starting Step 19 E2E Orchestration Pipeline execution...")

        exec_mode_raw = getattr(request, "execution_mode", None) or getattr(request, "executionMode", None) or "local"
        exec_mode = str(exec_mode_raw).lower().strip()
        if exec_mode in ["online", "mistral", "gemini"]:
            exec_mode = "online"
        else:
            exec_mode = "local"

        # --- STEP 15: ADAPTIVE DECISION ENGINE ---
        decision_req = DecisionRequest(
            text=request.prompt,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            execution_mode=exec_mode
        )
        decision_res = self.decision_engine.decide(decision_req)
        selected_model = decision_res.selected_model

        # --- UNCONFIGURED / NO EXECUTABLE MODEL PATH ---
        if not selected_model:
            logger.info("Decision engine selected no candidate model (unconfigured environment). Proceeding through unconfigured pipeline path...")
            gen_res = ResponseGenerationResponse(
                success=False,
                model_id=None,
                provider=None,
                generated_text=None,
                finish_reason=None,
                latency_ms=0.0,
                usage=None,
                execution_status="not_configured",
                error_message="No executable LLM candidate model is configured in environment settings."
            )

            ver_req = VerificationRequest(
                prompt=request.prompt,
                selected_model="none",
                generated_text=None,
                generation_success=False,
                execution_status="not_configured"
            )
            ver_res = self.response_verifier.verify_response(ver_req)

            reward_req = RewardComputeRequest(
                prompt=request.prompt,
                selected_model="none",
                winning_score=0.0,
                execution_success=False,
                execution_status="not_configured",
                generated_text=None,
                verification_status=ver_res.verification_status,
                verified=False,
                response_present=False,
                structural_quality_score=0.0,
                completeness_score=0.0,
                relevance_score=0.0,
                factual_verification_status="not_verified"
            )
            reward_res = self.reward_signal.compute_reward(reward_req)

            t1 = time.perf_counter()
            total_latency_ms = round((t1 - t0) * 1000, 2)

            orchestration_response = OrchestrationResponse(
                success=False,
                prompt=request.prompt,
                selected_model=None,
                decision_score=0.0,
                decision=decision_res,
                generation=gen_res,
                verification=ver_res,
                reward=reward_res,
                pipeline_latency_ms=total_latency_ms
            )

            # --- STEP 20: STRUCTURED TRANSITION LOGGING ---
            try:
                self.experience_buffer.record_from_orchestration(orchestration_response)
            except Exception as e:
                logger.warning(f"Failed to record experience in Step 20: {str(e)}")

            return orchestration_response

        # --- STEP 16: RESPONSE GENERATOR ---
        gen_req = ResponseGenerationRequest(
            prompt=request.prompt,
            selected_model=selected_model,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            execution_mode=exec_mode
        )
        gen_res = self.response_generator.generate_response(gen_req)

        # --- STEP 17: RESPONSE VERIFIER ---
        ver_req = VerificationRequest(
            prompt=request.prompt,
            selected_model=selected_model,
            generated_text=gen_res.generated_text,
            generation_success=gen_res.success,
            execution_status=gen_res.execution_status
        )
        ver_res = self.response_verifier.verify_response(ver_req)

        # --- STEP 18: REWARD SIGNAL ---
        reward_req = RewardComputeRequest(
            prompt=request.prompt,
            selected_model=selected_model,
            winning_score=decision_res.decision_score,
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
        reward_res = self.reward_signal.compute_reward(reward_req)

        t1 = time.perf_counter()
        total_latency_ms = round((t1 - t0) * 1000, 2)

        orchestration_response = OrchestrationResponse(
            success=gen_res.success,
            prompt=request.prompt,
            selected_model=selected_model,
            decision_score=decision_res.decision_score,
            decision=decision_res,
            generation=gen_res,
            verification=ver_res,
            reward=reward_res,
            pipeline_latency_ms=total_latency_ms
        )

        # --- STEP 20: STRUCTURED TRANSITION LOGGING ---
        try:
            self.experience_buffer.record_from_orchestration(orchestration_response)
        except Exception as e:
            logger.warning(f"Failed to record experience in Step 20: {str(e)}")

        return orchestration_response

    def run_pipeline_stream(self, request: OrchestrationRequest):
        import json
        import uuid
        t0 = time.perf_counter()

        run_id = request.run_id or f"run_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"

        def emit(evt: dict) -> str:
            evt["run_id"] = run_id
            return f"data: {json.dumps(evt)}\n\n"

        if not request.prompt or not request.prompt.strip():
            yield emit({"stage": "input_processing", "status": "failed", "error": "Prompt text cannot be empty."})
            return

        exec_mode_raw = getattr(request, "execution_mode", None) or getattr(request, "executionMode", None) or "local"
        exec_mode = str(exec_mode_raw).lower().strip()
        if exec_mode in ["online", "mistral", "gemini"]:
            exec_mode = "online"
        else:
            exec_mode = "local"

        logger.info(f"[RUN START] run_id={run_id} | prompt=\"{request.prompt}\" | execution_mode={exec_mode}")

        # Stage 1: Input Processing
        yield emit({"stage": "input_processing", "status": "running", "message": f"Validating prompt input processor... [{run_id}]"})
        yield emit({"stage": "input_processing", "status": "completed", "message": "Prompt input validated successfully."})

        # Stage 2: BGE-M3 Embedding
        yield emit({"stage": "embedding", "status": "running", "message": "Generating 1024D BGE-M3 semantic embedding..."})
        
        decision_req = DecisionRequest(
            text=request.prompt,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            execution_mode=exec_mode
        )
        decision_res = self.decision_engine.decide(decision_req)
        
        yield emit({"stage": "embedding", "status": "completed", "message": "BGE-M3 embedding generated (1024D).", "metadata": {"embedding_model": "BAAI/bge-m3"}})
        
        # Stage 3: Intent & Complexity Analysis
        yield emit({"stage": "intent_analysis", "status": "running", "message": "Classifying intent and analyzing multi-factor complexity..."})

        intent_info = decision_res.intent_info or {}
        complexity_info = decision_res.complexity_info or {}
        intent_name = intent_info.get("intent", "general_qa")
        complexity_level = complexity_info.get("complexity_level", "medium")

        yield emit({
            "stage": "intent_analysis",
            "status": "completed",
            "message": f"Intent: {intent_name} | Complexity: {complexity_level}",
            "metadata": {
                "intent": intent_info.get("intent"),
                "is_ambiguous": intent_info.get("is_ambiguous", False),
                "complexity_level": complexity_info.get("complexity_level"),
            },
        })

        # Stage 4: Decision Engine
        selected_model = decision_res.selected_model
        yield emit({"stage": "adaptive_decision", "status": "running", "message": f"Evaluating candidate models via {decision_res.policy}..."})

        yield emit({
            "stage": "adaptive_decision",
            "status": "completed",
            "message": f"Model {selected_model} selected (Score: {(decision_res.decision_score * 100):.1f}%).",
            "metadata": {
                "selected_model": selected_model,
                "score": decision_res.decision_score,
                "execution_mode": exec_mode,
                "candidates": [c.dict() for c in decision_res.candidates] if decision_res.candidates else []
            },
            "decision": decision_res.dict()
        })

        # Stage 5: Inference Execution (with BaselineAdaptivePolicy re-evaluation fallback)
        stage_5_key = "online_inference" if exec_mode == "online" else "local_inference"

        if not selected_model:
            yield emit({
                "stage": stage_5_key,
                "status": "failed",
                "error": "No executable LLM candidate model available.",
                "metadata": {
                    "execution_mode": exec_mode,
                    "provider": "Unknown",
                    "model": "none"
                }
            })
            return

        current_model = selected_model
        excluded_models = []

        gen_res = None
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts and current_model:
            attempt += 1
            if exec_mode == "online":
                evt5_run = {
                    "stage": "online_inference",
                    "status": "running",
                    "message": f"Executing Cloud API inference via Baseline Policy selected model ({current_model})...",
                    "metadata": {
                        "execution_mode": "online",
                        "provider": getattr(gen_res, "provider", None) or "Online API Pool",
                        "model": current_model
                    }
                }
            else:
                evt5_run = {
                    "stage": "local_inference",
                    "status": "running",
                    "message": f"Executing local Ollama inference ({current_model})...",
                    "metadata": {
                        "execution_mode": "local",
                        "provider": "Local Ollama",
                        "model": current_model
                    }
                }
            yield emit(evt5_run)

            logger.info(f"[RUN DISPATCH] run_id={run_id} | attempt={attempt} | model={current_model} | prompt=\"{request.prompt}\"")

            gen_req = ResponseGenerationRequest(
                prompt=request.prompt,
                selected_model=current_model,
                system_instruction=request.system_instruction,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                execution_mode=exec_mode
            )
            gen_res = self.response_generator.generate_response(gen_req)

            if gen_res.success and gen_res.generated_text:
                selected_model = current_model
                logger.info(f"[RUN GENERATION SUCCESS] run_id={run_id} | model={current_model} | text_len={len(gen_res.generated_text)}")
                break

            logger.warning(f"[RUN FAILURE] run_id={run_id} | candidate='{current_model}' failed. Excluding and re-evaluating...")
            excluded_models.append(current_model)

            # Re-evaluate remaining candidates through BaselineAdaptivePolicy
            try:
                dec_req_fb = DecisionRequest(
                    text=request.prompt,
                    execution_mode=exec_mode,
                    excluded_models=excluded_models
                )
                fb_dec = self.decision_engine.decide(dec_req_fb)
                if fb_dec.selected_model and fb_dec.selected_model not in excluded_models:
                    current_model = fb_dec.selected_model
                    decision_res = fb_dec
                    logger.info(f"[RUN FALLBACK RE-SELECT] run_id={run_id} | re_selected='{current_model}'")
                else:
                    break
            except Exception as fb_err:
                logger.error(f"Fallback re-evaluation error: {str(fb_err)}")
                break

        provider_label = gen_res.provider if (gen_res and gen_res.provider) else ("Online Cloud API" if exec_mode == "online" else "Local Ollama")
        yield emit({
            "stage": stage_5_key,
            "status": "completed",
            "message": f"{provider_label} inference completed in {gen_res.latency_ms:.1f}ms." if gen_res else "Inference completed.",
            "metadata": {
                "execution_mode": exec_mode,
                "provider": provider_label,
                "model": gen_res.model_id if gen_res else selected_model,
                "model_id": gen_res.model_id if gen_res else selected_model,
                "latency_ms": gen_res.latency_ms if gen_res else 0.0
            }
        })

        # Stage 6: Response Verification
        yield emit({"stage": "response_verifier", "status": "running", "message": "Running structural and keyword relevance verification..."})

        ver_req = VerificationRequest(
            prompt=request.prompt,
            selected_model=selected_model,
            generated_text=gen_res.generated_text if gen_res else "",
            generation_success=gen_res.success if gen_res else False,
            execution_status=gen_res.execution_status if gen_res else "failed"
        )
        ver_res = self.response_verifier.verify_response(ver_req)

        yield emit({
            "stage": "response_verifier",
            "status": "completed",
            "message": f"Verification {ver_res.verification_status} (Verified: {ver_res.verified}).",
            "metadata": {
                "verified": ver_res.verified,
                "status": ver_res.verification_status
            }
        })

        # Stage 7: Reward Signal
        yield emit({"stage": "reward_signal", "status": "running", "message": "Computing Step 18 deterministic reward signal..."})

        reward_req = RewardComputeRequest(
            prompt=request.prompt,
            selected_model=selected_model,
            winning_score=decision_res.decision_score,
            execution_success=gen_res.success if gen_res else False,
            execution_status=gen_res.execution_status if gen_res else "failed",
            generated_text=gen_res.generated_text if gen_res else "",
            verification_status=ver_res.verification_status,
            verified=ver_res.verified,
            response_present=ver_res.response_present,
            structural_quality_score=ver_res.structural_quality_score,
            completeness_score=ver_res.completeness_score,
            relevance_score=ver_res.relevance_score,
            factual_verification_status=ver_res.factual_verification_status
        )
        reward_res = self.reward_signal.compute_reward(reward_req)

        yield emit({
            "stage": "reward_signal",
            "status": "completed",
            "message": f"Step 18 Reward computed: {reward_res.reward:.4f}.",
            "metadata": {"reward": reward_res.reward}
        })

        t1 = time.perf_counter()
        total_latency_ms = round((t1 - t0) * 1000, 2)

        orchestration_response = OrchestrationResponse(
            run_id=run_id,
            success=gen_res.success if gen_res else False,
            prompt=request.prompt,
            selected_model=selected_model,
            decision_score=decision_res.decision_score,
            decision=decision_res,
            generation=gen_res,
            verification=ver_res,
            reward=reward_res,
            pipeline_latency_ms=total_latency_ms
        )

        # Stage 8: Step 20 Replay Buffer
        yield emit({"stage": "experience_replay", "status": "running", "message": "Recording transition into 12D experience replay buffer..."})

        try:
            self.experience_buffer.record_from_orchestration(orchestration_response)
            yield emit({"stage": "experience_replay", "status": "completed", "message": "12D transition recorded into experience buffer."})
        except Exception as e:
            logger.error(f"Failed to record transition into experience buffer: {str(e)}")
            yield emit({"stage": "experience_replay", "status": "failed", "error": str(e)})

        # Final Response Delivery Payload
        logger.info(f"[RUN FINAL] run_id={run_id} | prompt=\"{request.prompt}\" | model={selected_model} | text_len={len(gen_res.generated_text if gen_res and gen_res.generated_text else '')}")
        yield emit({
            "stage": "final_response",
            "status": "completed",
            "message": f"E2E Orchestration Pipeline completed in {total_latency_ms:.1f}ms.",
            "payload": orchestration_response.dict()
        })
