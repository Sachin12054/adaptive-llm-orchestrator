import time
import json
import uuid
import asyncio
import logging
from typing import Optional

from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.response_generator import ResponseGenerator
from app.services.response_verifier import ResponseVerifier
from app.services.reward_signal import RewardSignal
from app.services.experience_buffer import ExperienceBufferService
from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer
from app.services.complex.dynamic_decomposer import DynamicTaskDecomposer
from app.services.provider_failover import RequestProviderTracker, classify_failure

from app.schemas.decision import DecisionRequest
from app.schemas.response import ResponseGenerationRequest, ResponseGenerationResponse
from app.schemas.verification import VerificationRequest
from app.schemas.reward import RewardComputeRequest
from app.schemas.provider import TokenUsage
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
        experience_buffer: Optional[ExperienceBufferService] = None,
        complex_decomposer: Optional[ComplexTaskDecomposer] = None
    ):
        self.decision_engine = decision_engine or AdaptiveDecisionEngine()
        self.response_generator = response_generator or ResponseGenerator()
        self.response_verifier = response_verifier or ResponseVerifier()
        self.reward_signal = reward_signal or RewardSignal()
        self.experience_buffer = experience_buffer or ExperienceBufferService()
        self.complex_decomposer = complex_decomposer or ComplexTaskDecomposer(
            dynamic_decomposer=DynamicTaskDecomposer(
                decision_engine=self.decision_engine,
                experience_buffer=self.experience_buffer
            )
        )

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

        complexity_info = decision_res.complexity_info or {}
        cmplx_score = complexity_info.get("complexity_score", 0.0)
        cmplx_level = complexity_info.get("complexity_level", "medium")

        # --- COMPLEX TASK DECOMPOSITION CHECK ---
        if self.complex_decomposer.is_complex_prompt(request.prompt, cmplx_score, cmplx_level):
            logger.info(f"Complex multi-task query detected (score={cmplx_score:.4f}, level={cmplx_level}). Executing parallel task decomposition...")
            
            # Run async execution plan
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            complex_plan = loop.run_until_complete(
                self.complex_decomposer.execute_complex_plan_async(
                    prompt=request.prompt,
                    execution_mode=exec_mode
                )
            )

            usage = None
            if complex_plan.total_tokens > 0 or complex_plan.total_input_tokens > 0 or complex_plan.total_output_tokens > 0:
                from app.schemas.provider import TokenUsage
                usage = TokenUsage(
                    input_tokens=complex_plan.total_input_tokens,
                    output_tokens=complex_plan.total_output_tokens,
                    total_tokens=complex_plan.total_tokens or (complex_plan.total_input_tokens + complex_plan.total_output_tokens)
                )

            actual_executed_models = [
                str(getattr(st, "assigned_model", "")) for st in (complex_plan.subtasks or [])
                if getattr(st, "execution_success", False) and getattr(st, "assigned_model", None)
            ]
            actual_executed_providers = [
                str(getattr(st, "provider", "")) for st in (complex_plan.subtasks or [])
                if getattr(st, "execution_success", False) and getattr(st, "provider", None)
            ]
            final_actual_model = actual_executed_models[0] if len(actual_executed_models) == 1 else ("multiple_models" if actual_executed_models else selected_model)
            final_actual_provider = actual_executed_providers[0] if len(actual_executed_providers) == 1 else ("multiple_providers" if actual_executed_providers else "Multi-LLM Parallel Execution Engine")
            failover_used = any(bool(getattr(st, "failover_used", False)) for st in (complex_plan.subtasks or []))

            gen_res = ResponseGenerationResponse(
                success=complex_plan.execution_success,
                model_id=final_actual_model,
                provider=final_actual_provider,
                generated_text=complex_plan.aggregated_response,
                finish_reason="STOP",
                latency_ms=complex_plan.total_execution_latency_ms,
                usage=usage,
                cost=complex_plan.total_workflow_cost,
                cost_currency=complex_plan.cost_currency,
                cost_source=complex_plan.cost_source,
                execution_status="completed" if complex_plan.execution_success else "failed",
                initial_model=selected_model,
                failover_used=failover_used,
                attempts=max(1, sum(int(getattr(st, "attempts", 1) or 1) for st in (complex_plan.subtasks or [])))
            )

            ver_req = VerificationRequest(
                prompt=request.prompt,
                selected_model=selected_model or "complex_multi_llm",
                generated_text=complex_plan.aggregated_response,
                generation_success=complex_plan.execution_success,
                execution_status=gen_res.execution_status
            )
            ver_res = self.response_verifier.verify_response(ver_req)

            reward_req = RewardComputeRequest(
                prompt=request.prompt,
                selected_model=selected_model or "complex_multi_llm",
                winning_score=decision_res.decision_score,
                execution_success=complex_plan.execution_success,
                execution_status=gen_res.execution_status,
                generated_text=complex_plan.aggregated_response,
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
                success=complex_plan.execution_success,
                prompt=request.prompt,
                selected_model=selected_model,
                decision_score=decision_res.decision_score,
                decision=decision_res,
                generation=gen_res,
                verification=ver_res,
                reward=reward_res,
                pipeline_latency_ms=total_latency_ms,
                complex_plan=complex_plan
            )

            try:
                self.experience_buffer.record_from_orchestration(orchestration_response)
            except Exception as e:
                logger.warning(f"Failed to record experience in Step 20: {str(e)}")

            return orchestration_response

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

            try:
                self.experience_buffer.record_from_orchestration(orchestration_response)
            except Exception as e:
                logger.warning(f"Failed to record experience in Step 20: {str(e)}")

            return orchestration_response

        # --- SINGLE-MODEL STANDARD RESPONSE GENERATOR WITH FAILOVER ---
        logger.info(f"BASELINE SELECTION: selected_model={selected_model} | execution_mode={exec_mode}")
        current_model = selected_model
        excluded_models = []
        gen_res = None
        max_attempts = 4
        attempt = 0
        provider_tracker = RequestProviderTracker()

        while attempt < max_attempts and current_model:
            attempt += 1
            gen_req = ResponseGenerationRequest(
                prompt=request.prompt,
                selected_model=current_model,
                system_instruction=request.system_instruction,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                execution_mode=exec_mode
            )
            gen_res = self.response_generator.generate_response(gen_req)
            if gen_res.success:
                selected_model = current_model
                break

            failure_type = classify_failure(gen_res.error_message)
            if failure_type == "non_retryable":
                break

            curr_meta = self.decision_engine.model_registry.get_model(current_model)
            if curr_meta:
                provider_tracker.mark_provider_exhausted(curr_meta.provider, failure_type)

            excluded_models.append(current_model)
            re_req = DecisionRequest(
                text=request.prompt,
                system_instruction=request.system_instruction,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                execution_mode=exec_mode,
                excluded_models=excluded_models,
                excluded_providers=list(provider_tracker.exhausted_providers.keys())
            )
            try:
                next_dec = self.decision_engine.decide(re_req)
                current_model = next_dec.selected_model
                if not current_model or current_model in excluded_models:
                    break
            except Exception:
                break

        ver_req = VerificationRequest(
            prompt=request.prompt,
            selected_model=selected_model,
            generated_text=gen_res.generated_text,
            generation_success=gen_res.success,
            execution_status=gen_res.execution_status
        )
        ver_res = self.response_verifier.verify_response(ver_req)

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

        try:
            self.experience_buffer.record_from_orchestration(orchestration_response)
        except Exception as e:
            logger.warning(f"Failed to record experience in Step 20: {str(e)}")

        return orchestration_response

    async def run_pipeline_stream(self, request: OrchestrationRequest):
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
        cmplx_score = complexity_info.get("complexity_score", 0.0)

        yield emit({
            "stage": "intent_analysis",
            "status": "completed",
            "message": f"Intent: {intent_name} | Complexity: {complexity_level}",
            "metadata": {
                "intent": intent_name,
                "is_ambiguous": intent_info.get("is_ambiguous", False),
                "complexity_level": complexity_level,
                "complexity_score": cmplx_score
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

        # --- CHECK IF COMPLEX DECOMPOSITION IS TRIGGERED ---
        is_complex = self.complex_decomposer.is_complex_prompt(request.prompt, cmplx_score, complexity_level)

        if is_complex:
            yield emit({
                "stage": "decomposition_started",
                "status": "running",
                "message": "Complex multi-objective query detected. Initializing dynamic task decomposition & DAG scheduler..."
            })

            # Queue for collecting async events inside task scheduler
            event_queue = asyncio.Queue()

            async def queue_event(evt):
                await event_queue.put(evt)

            # Launch task execution coroutine
            task_exec_coro = self.complex_decomposer.execute_complex_plan_async(
                prompt=request.prompt,
                execution_mode=exec_mode,
                event_callback=queue_event
            )

            task_future = asyncio.create_task(task_exec_coro)

            # Stream intermediate task execution events
            while not task_future.done() or not event_queue.empty():
                try:
                    evt_data = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    yield emit(evt_data)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.05)

            try:
                complex_plan = await task_future
            except Exception as exc:
                logger.error(f"[PIPELINE EXCEPTION] Complex plan execution error: {str(exc)}", exc_info=True)
                yield emit({
                    "stage": "complex_execution",
                    "status": "failed",
                    "message": f"Complex task execution encountered an exception: {str(exc)}",
                    "metadata": {"error": str(exc)}
                })
                from app.schemas.complex import ComplexTaskPlan
                complex_plan = ComplexTaskPlan(
                    is_complex=True,
                    original_prompt=request.prompt,
                    subtasks=[],
                    execution_levels=[],
                    total_subtasks=0,
                    plan_latency_ms=0.0,
                    aggregated_response=f"Complex task execution encountered an error: {str(exc)}",
                    execution_success=False,
                    total_execution_latency_ms=0.0
                )

            gen_res = ResponseGenerationResponse(
                success=complex_plan.execution_success,
                model_id=complex_plan.final_model,
                provider=complex_plan.final_provider,
                generated_text=complex_plan.aggregated_response,
                finish_reason="STOP",
                latency_ms=complex_plan.total_execution_latency_ms,
                usage=TokenUsage(
                    input_tokens=complex_plan.total_input_tokens,
                    output_tokens=complex_plan.total_output_tokens,
                    total_tokens=complex_plan.total_tokens,
                ),
                cost=complex_plan.total_workflow_cost,
                cost_currency=complex_plan.cost_currency,
                cost_source=complex_plan.cost_source,
                initial_model=selected_model,
                failover_used=complex_plan.failover_used,
                attempts=max(1, sum(int(getattr(st, "attempts", 1) or 1) for st in complex_plan.subtasks)),
                execution_status="completed" if complex_plan.execution_success else "failed"
            )

            # Response Verification
            ver_req = VerificationRequest(
                prompt=request.prompt,
                selected_model=selected_model or "complex_multi_llm",
                generated_text=complex_plan.aggregated_response,
                generation_success=complex_plan.execution_success,
                execution_status=gen_res.execution_status
            )
            ver_res = self.response_verifier.verify_response(ver_req)

            # Reward Signal
            reward_req = RewardComputeRequest(
                prompt=request.prompt,
                selected_model=selected_model or "complex_multi_llm",
                winning_score=decision_res.decision_score,
                execution_success=complex_plan.execution_success,
                execution_status=gen_res.execution_status,
                generated_text=complex_plan.aggregated_response,
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
                run_id=run_id,
                success=complex_plan.execution_success,
                prompt=request.prompt,
                selected_model=selected_model,
                decision_score=decision_res.decision_score,
                decision=decision_res,
                generation=gen_res,
                verification=ver_res,
                reward=reward_res,
                pipeline_latency_ms=total_latency_ms,
                complex_plan=complex_plan
            )

            try:
                self.experience_buffer.record_from_orchestration(orchestration_response)
            except Exception as e:
                logger.warning(f"Failed to record experience in Step 20: {str(e)}")

            yield emit({
                "stage": "final_response",
                "status": "completed",
                "message": f"Complex Multi-LLM Orchestration completed in {total_latency_ms:.1f}ms.",
                "metadata": {
                    "selected_model": selected_model,
                    "provider": complex_plan.final_provider,
                    "actual_models": complex_plan.actual_models,
                    "actual_providers": complex_plan.actual_providers,
                    "failover_used": complex_plan.failover_used,
                    "total_workflow_cost": complex_plan.total_workflow_cost,
                    "execution_mode": exec_mode,
                    "pipeline_latency_ms": total_latency_ms,
                    "is_complex": True
                },
                "payload": orchestration_response.dict()
            })
            return

        # --- STANDARD SINGLE-MODEL STREAMING EXECUTION ---
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

        initial_meta = self.decision_engine.model_registry.get_model(selected_model) if selected_model else None
        initial_provider = initial_meta.provider if initial_meta else ("Online API" if exec_mode == "online" else "Local Ollama")
        logger.info(f"BASELINE SELECTION: selected_model={selected_model} | provider={initial_provider}")

        gen_res = None
        max_attempts = 4
        attempt = 0
        provider_tracker = RequestProviderTracker()
        attempts_detail = []
        initial_selected_model = selected_model

        while attempt < max_attempts and current_model:
            attempt += 1
            curr_meta = self.decision_engine.model_registry.get_model(current_model)
            curr_provider = curr_meta.provider if curr_meta else ("Online API" if exec_mode == "online" else "Local Ollama")

            if provider_tracker.is_provider_exhausted(curr_provider):
                logger.info(f"[PROVIDER COOLDOWN] Provider '{curr_provider}' is exhausted for this request. Skipping {current_model}.")
                excluded_models.append(current_model)
                try:
                    dec_req_fb = DecisionRequest(
                        text=request.prompt,
                        execution_mode=exec_mode,
                        excluded_models=excluded_models,
                        excluded_providers=provider_tracker.get_exhausted_list()
                    )
                    fb_dec = self.decision_engine.decide(dec_req_fb)
                    if fb_dec.selected_model and fb_dec.selected_model not in excluded_models:
                        current_model = fb_dec.selected_model
                        decision_res = fb_dec
                        continue
                    else:
                        break
                except Exception:
                    break

            t_att_start = time.perf_counter()
            evt5_run = {
                "stage": stage_5_key,
                "status": "running",
                "message": f"Attempt {attempt}: Executing inference via {current_model} ({curr_provider})...",
                "metadata": {
                    "execution_mode": exec_mode,
                    "provider": curr_provider,
                    "model": current_model,
                    "attempt": attempt
                }
            }
            yield emit(evt5_run)

            gen_req = ResponseGenerationRequest(
                prompt=request.prompt,
                selected_model=current_model,
                system_instruction=request.system_instruction,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                execution_mode=exec_mode
            )
            gen_res = self.response_generator.generate_response(gen_req)
            t_att_end = time.perf_counter()
            att_latency_ms = round((t_att_end - t_att_start) * 1000, 2)

            if gen_res.success and gen_res.generated_text and gen_res.generated_text.strip():
                selected_model = current_model
                attempts_detail.append({
                    "attempt_number": attempt,
                    "model": current_model,
                    "provider": gen_res.provider or curr_provider,
                    "status": "success",
                    "error": None,
                    "latency_ms": att_latency_ms,
                    "timestamp": time.time()
                })
                gen_res.initial_model = initial_selected_model
                gen_res.failover_used = (attempt > 1 or current_model != initial_selected_model)
                gen_res.attempts = attempt
                gen_res.attempts_detail = attempts_detail
                break

            failed_model = current_model
            failed_provider = curr_provider
            err_details = gen_res.error_message if (gen_res and gen_res.error_message) else "Inference failed"
            is_retryable, failure_type = classify_failure(err_details)
            excluded_models.append(failed_model)

            attempts_detail.append({
                "attempt_number": attempt,
                "model": current_model,
                "provider": curr_provider,
                "status": "failed",
                "failure_type": failure_type,
                "error": err_details,
                "latency_ms": att_latency_ms,
                "timestamp": time.time()
            })

            if failure_type in ["quota_exhausted", "rate_limit"]:
                provider_tracker.mark_provider_exhausted(curr_provider, failure_type)

            if not is_retryable:
                yield emit({
                    "stage": stage_5_key,
                    "status": "failed",
                    "message": f"{failed_model} failed with non-retryable error ({err_details})",
                    "metadata": {
                        "failed_model": failed_model,
                        "error": err_details,
                        "execution_mode": exec_mode
                    }
                })
                break

            try:
                dec_req_fb = DecisionRequest(
                    text=request.prompt,
                    execution_mode=exec_mode,
                    excluded_models=excluded_models,
                    excluded_providers=provider_tracker.get_exhausted_list()
                )
                fb_dec = self.decision_engine.decide(dec_req_fb)
                if fb_dec.selected_model and fb_dec.selected_model not in excluded_models:
                    current_model = fb_dec.selected_model
                    decision_res = fb_dec
                    yield emit({
                        "stage": stage_5_key,
                        "status": "fallback",
                        "message": f"[FAILOVER] {failed_model} failed ({failure_type}); falling back to {current_model}",
                        "metadata": {
                            "failed_model": failed_model,
                            "error": err_details,
                            "failure_type": failure_type,
                            "re_selected_model": current_model,
                            "execution_mode": exec_mode,
                            "attempt": attempt
                        }
                    })
                else:
                    yield emit({
                        "stage": stage_5_key,
                        "status": "failed",
                        "message": f"{failed_model} failed ({err_details}); no remaining candidate models",
                        "metadata": {
                            "failed_model": failed_model,
                            "error": err_details,
                            "execution_mode": exec_mode
                        }
                    })
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

        yield emit({"stage": "experience_replay", "status": "running", "message": "Recording transition into 12D experience replay buffer..."})

        try:
            self.experience_buffer.record_from_orchestration(orchestration_response)
            yield emit({"stage": "experience_replay", "status": "completed", "message": "12D transition recorded into experience buffer."})
        except Exception as e:
            logger.error(f"Failed to record transition into experience buffer: {str(e)}")
            yield emit({"stage": "experience_replay", "status": "failed", "error": str(e)})

        yield emit({
            "stage": "final_response",
            "status": "completed",
            "message": f"E2E Orchestration Pipeline completed in {total_latency_ms:.1f}ms.",
            "metadata": {
                "selected_model": selected_model,
                "provider": provider_label,
                "execution_mode": exec_mode,
                "pipeline_latency_ms": total_latency_ms
            },
            "payload": orchestration_response.dict()
        })
