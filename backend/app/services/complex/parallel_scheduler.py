import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Set

from app.schemas.complex import SubTask
from app.services.response_generator import ResponseGenerator
from app.schemas.response import ResponseGenerationRequest
from app.schemas.decision import DecisionRequest
from app.services.provider_failover import classify_failure, RequestProviderTracker
from app.services.complex.task_response_validator import TaskResponseValidator

logger = logging.getLogger("orchestrator")

class ParallelTaskScheduler:
    """
    Async Adaptive Concurrency DAG Task Scheduler with Dynamic Provider Failover & Task Immutability.
    - Manages task execution lifecycle: PENDING -> READY -> RUNNING -> COMPLETED / FAILED.
    - Event-Driven Async DAG Scheduling: Immediately unlocks and dispatches dependent tasks as soon as prerequisites complete.
    - Independent Provider Semaphores: Cloud providers (Gemini, Mistral, Groq, OpenRouter) run concurrently with Local Ollama.
    - Adaptive Local Concurrency: Scales local Ollama concurrency dynamically (1–3 workers) based on CPU/RAM/GPU telemetry.
    - Global Prompt Context & Hash Immutability: Injects original user prompt as global context and asserts SHA-256 prompt hash before every attempt.
    - Generic Domain-Consistency Validation: Verifies model response relevance against original user request domain; rejects wrong-domain outputs.
    - Local Timeout & Token Budgeting: Enforces 60s timeout on local Ollama calls and applies category output token budgets.
    """

    def __init__(
        self,
        response_generator: Optional[ResponseGenerator] = None,
        max_concurrency: int = 7,
        decision_engine: Optional[Any] = None,
        resource_analyzer: Optional[Any] = None
    ):
        self.response_generator = response_generator or ResponseGenerator()
        self.max_concurrency = max_concurrency
        self._decision_engine = decision_engine
        self._resource_analyzer = resource_analyzer

        # Independent Cloud Provider Semaphores
        self.cloud_semaphores: Dict[str, asyncio.Semaphore] = {
            "Google Gemini API": asyncio.Semaphore(4),
            "Mistral API": asyncio.Semaphore(4),
            "Groq API": asyncio.Semaphore(4),
            "OpenRouter": asyncio.Semaphore(4),
            "default_cloud": asyncio.Semaphore(4)
        }
        # Local Ollama semaphore adaptively scaled per request
        self.local_semaphore = asyncio.Semaphore(2)

    @property
    def decision_engine(self):
        if self._decision_engine is None:
            from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
            self._decision_engine = AdaptiveDecisionEngine()
        return self._decision_engine

    @property
    def resource_analyzer(self):
        if self._resource_analyzer is None:
            from app.services.resource_analyzer import ResourceAnalyzer
            self._resource_analyzer = ResourceAnalyzer()
        return self._resource_analyzer

    def _get_adaptive_local_concurrency(self) -> int:
        """
        Calculates optimal local Ollama concurrency based on CPU, RAM, and GPU telemetry.
        Decouples RUNNABLE TASK COUNT from active local model generations to prevent CPU thrashing.
        """
        try:
            snapshot = self.resource_analyzer.get_resource_snapshot()
            cpu_util = snapshot.cpu.utilization_percent
            avail_ram = snapshot.memory.available_gb
            gpu_avail = snapshot.gpu.available

            if gpu_avail:
                return min(3, self.max_concurrency)

            if cpu_util > 80.0 or avail_ram < 4.0:
                return 1
            elif cpu_util > 60.0 or avail_ram < 6.0:
                return 2
            else:
                return min(3, self.max_concurrency)
        except Exception:
            return 2

    def _get_provider_semaphore(self, provider_name: Optional[str], execution_mode: str) -> asyncio.Semaphore:
        """
        Returns the appropriate provider semaphore for local vs cloud execution.
        """
        if execution_mode.lower() == "local" or (provider_name and "ollama" in provider_name.lower()):
            return self.local_semaphore
        
        if provider_name in self.cloud_semaphores:
            return self.cloud_semaphores[provider_name]
        
        for key, sem in self.cloud_semaphores.items():
            if provider_name and key.lower() in provider_name.lower():
                return sem
                
        return self.cloud_semaphores["default_cloud"]

    async def execute_plan_async(
        self,
        subtasks: List[SubTask],
        execution_levels: List[List[str]],
        execution_mode: str = "local",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        provider_tracker: Optional[RequestProviderTracker] = None
    ) -> List[SubTask]:
        """
        Executes decomposed subtasks using an Event-Driven Async DAG Queue.
        Dispatches newly unblocked tasks immediately as soon as prerequisites finish.
        Concurrently executes independent cloud and local tasks subject to provider semaphores.
        """
        if not subtasks:
            return []

        if provider_tracker is None:
            provider_tracker = RequestProviderTracker()

        # Update adaptive local semaphore based on hardware load
        local_concurrency = self._get_adaptive_local_concurrency()
        self.local_semaphore = asyncio.Semaphore(local_concurrency)
        logger.info(f"[SCHEDULER INIT] execution_mode={execution_mode} | adaptive_local_concurrency={local_concurrency} | max_concurrency={self.max_concurrency}")

        task_map: Dict[str, SubTask] = {t.task_id: t for t in subtasks}
        completed_outputs: Dict[str, str] = {}
        completed_ids: Set[str] = set()
        failed_ids: Set[str] = set()
        running_tasks: Dict[asyncio.Task, str] = {}

        total_tasks = len(subtasks)

        while len(completed_ids) + len(failed_ids) < total_tasks:
            # 1. Find all runnable subtasks (all prerequisites are in completed_ids or failed_ids)
            runnable = [
                t for t in subtasks 
                if t.status in ["PENDING", "READY"] 
                and t.task_id not in running_tasks.values()
                and all(dep in completed_ids or dep in failed_ids for dep in t.dependencies)
            ]

            # 2. Launch background task coroutines for newly runnable subtasks
            for t in runnable:
                t.status = "RUNNING"
                if event_callback:
                    await self._emit_async(event_callback, {
                        "stage": "complex_execution",
                        "event_type": "task_ready",
                        "status": "ready",
                        "task_id": t.task_id,
                        "message": f"Subtask {t.task_id} ready and starting execution.",
                        "metadata": {
                            "task_id": t.task_id,
                            "model": t.assigned_model,
                            "provider": t.provider
                        }
                    })

                task_coro = self._execute_single_subtask(
                    subtask=t,
                    completed_outputs=completed_outputs,
                    execution_mode=execution_mode,
                    event_callback=event_callback,
                    provider_tracker=provider_tracker
                )
                async_task = asyncio.create_task(task_coro)
                running_tasks[async_task] = t.task_id

            if not running_tasks:
                # Deadlock safety check
                logger.warning(f"[SCHEDULER DEADLOCK] No active running tasks. Completed: {len(completed_ids)}, Failed: {len(failed_ids)}, Total: {total_tasks}")
                for t in subtasks:
                    if t.status in ["PENDING", "READY"]:
                        t.status = "FAILED"
                        t.error_message = "Task blocked due to unfulfilled dependency failures."
                        failed_ids.add(t.task_id)
                break

            # 3. Wait for the FIRST completed task (Early Dependency Unlocking)
            done, _ = await asyncio.wait(running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

            for finished_task in done:
                finished_task_id = running_tasks.pop(finished_task, None)
                try:
                    res_subtask = finished_task.result()
                    if res_subtask.execution_success and res_subtask.generated_text:
                        completed_ids.add(res_subtask.task_id)
                        completed_outputs[res_subtask.task_id] = res_subtask.generated_text
                    else:
                        failed_ids.add(res_subtask.task_id)
                except Exception as exc:
                    logger.error(f"[TASK EXCEPTION] task_id={finished_task_id} | error={str(exc)}")
                    if finished_task_id and finished_task_id in task_map:
                        st = task_map[finished_task_id]
                        st.status = "FAILED"
                        st.execution_success = False
                        st.error_message = str(exc)
                        failed_ids.add(st.task_id)

        return list(task_map.values())

    async def _execute_single_subtask(
        self,
        subtask: SubTask,
        completed_outputs: Dict[str, str],
        execution_mode: str = "local",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        provider_tracker: Optional[RequestProviderTracker] = None
    ) -> SubTask:
        # Initialize immutable task specification fields
        if not subtask.task_objective:
            subtask.task_objective = subtask.description
        if not subtask.prompt_hash:
            subtask.prompt_hash = TaskResponseValidator.compute_prompt_hash(
                subtask.task_id,
                subtask.original_user_prompt,
                subtask.task_objective
            )
        if not subtask.initial_model:
            subtask.initial_model = subtask.assigned_model

        subtask.status = "RUNNING"
        current_model = subtask.assigned_model
        attempted_models: Set[str] = set()
        subtask.attempts_detail = []
        subtask.failover_used = False

        # 1. Assert Prompt Immutability & SHA-256 Hash Integrity
        try:
            TaskResponseValidator.validate_prompt_integrity(
                expected_hash=subtask.prompt_hash,
                task_id=subtask.task_id,
                original_user_prompt=subtask.original_user_prompt,
                task_objective=subtask.task_objective
            )
        except ValueError as val_err:
            subtask.status = "FAILED"
            subtask.execution_success = False
            subtask.error_message = str(val_err)
            return subtask

        # 2. Construct Prompt with Global Request Context & Prerequisites
        global_ctx_header = (
            f"Global User Request Context: {subtask.original_user_prompt}\n\n"
            if subtask.original_user_prompt else ""
        )

        prereq_context = ""
        if subtask.dependencies:
            prereq_texts = []
            for dep_id in subtask.dependencies:
                if dep_id in completed_outputs:
                    prereq_texts.append(f"--- Context from Prerequisite {dep_id} ---\n{completed_outputs[dep_id]}")
                else:
                    prereq_texts.append(f"--- Context from Prerequisite {dep_id} ---\n[Output unavailable]")
            prereq_context = "\n\n".join(prereq_texts) + "\n\n"

        subtask_prompt = (
            f"{global_ctx_header}"
            f"{prereq_context}"
            f"Assigned Subtask Objective: {subtask.task_objective}\n\n"
            f"Instructions: You are solving ONLY this specific subtask objective as part of the overall request above. "
            f"Provide direct, concise, factual details specifically tailored to the domain in the global request context. "
            f"Do not produce generic schemas or off-target domain answers."
        )

        max_attempts = 4
        attempt = 0

        while attempt < max_attempts and current_model:
            attempt += 1
            subtask.attempts = attempt

            curr_meta = self.decision_engine.model_registry.get_model(current_model)
            curr_provider = curr_meta.provider if curr_meta else ("Local Ollama" if execution_mode == "local" else "Online API")

            logger.info(
                f"[ATTEMPT DISPATCH] task_id={subtask.task_id} | attempt={attempt} | "
                f"model={current_model} | provider={curr_provider} | prompt_hash={subtask.prompt_hash[:12]}..."
            )

            # Provider Quota Cooldown Check
            if provider_tracker and provider_tracker.is_provider_exhausted(curr_provider):
                logger.info(f"[PROVIDER COOLDOWN] Subtask {subtask.task_id}: Provider '{curr_provider}' is exhausted. Skipping {current_model}.")
                attempted_models.add(current_model)
                
                next_model = self._select_next_candidate(
                    subtask_description=subtask.task_objective,
                    category=subtask.category,
                    execution_mode=execution_mode,
                    attempted_models=attempted_models,
                    provider_tracker=provider_tracker
                )

                if next_model and next_model not in attempted_models:
                    if event_callback:
                        await self._emit_async(event_callback, {
                            "stage": "complex_execution",
                            "event_type": "provider_failover",
                            "status": "failover",
                            "task_id": subtask.task_id,
                            "message": f"[PROVIDER COOLDOWN] {curr_provider} exhausted; re-routing {subtask.task_id} from {current_model} to {next_model}",
                            "metadata": {
                                "task_id": subtask.task_id,
                                "primary_model": current_model,
                                "failure_reason": f"Provider {curr_provider} quota exhausted for request",
                                "fallback_model": next_model,
                                "prompt_hash": subtask.prompt_hash
                            }
                        })
                    current_model = next_model
                    continue
                else:
                    break

            t_attempt_start = time.perf_counter()

            if event_callback:
                await self._emit_async(event_callback, {
                    "stage": "complex_execution",
                    "event_type": "provider_attempt_started",
                    "status": "running",
                    "task_id": subtask.task_id,
                    "message": f"Attempt {attempt}: Executing {subtask.task_id} on {current_model} ({curr_provider})...",
                    "metadata": {
                        "task_id": subtask.task_id,
                        "attempt": attempt,
                        "model": current_model,
                        "provider": curr_provider,
                        "prompt_hash": subtask.prompt_hash
                    }
                })

            gen_req = ResponseGenerationRequest(
                prompt=subtask_prompt,
                selected_model=current_model,
                execution_mode=execution_mode,
                max_output_tokens=subtask.max_output_tokens
            )

            # Acquire Provider-Specific Semaphore
            provider_sem = self._get_provider_semaphore(curr_provider, execution_mode)

            try:
                async with provider_sem:
                    loop = asyncio.get_running_loop()
                    # Apply 60s timeout for local Ollama calls to prevent long generation hangs
                    if execution_mode.lower() == "local" or "ollama" in curr_provider.lower():
                        gen_res = await asyncio.wait_for(
                            loop.run_in_executor(None, self.response_generator.generate_response, gen_req),
                            timeout=60.0
                        )
                    else:
                        gen_res = await loop.run_in_executor(None, self.response_generator.generate_response, gen_req)

                t_attempt_end = time.perf_counter()
                att_latency_ms = round((t_attempt_end - t_attempt_start) * 1000, 2)

                if gen_res.success and gen_res.generated_text and gen_res.generated_text.strip():
                    # 3. Perform Generic Semantic Domain-Consistency Validation
                    is_valid, rel_score, rel_reason = TaskResponseValidator.validate_response_relevance(
                        task_id=subtask.task_id,
                        original_user_prompt=subtask.original_user_prompt,
                        task_objective=subtask.task_objective,
                        category=subtask.category,
                        response_text=gen_res.generated_text
                    )
                    subtask.relevance_score = rel_score

                    if is_valid:
                        # SUCCESS & RELEVANT!
                        subtask.execution_success = True
                        subtask.status = "COMPLETED"
                        subtask.actual_model = current_model
                        subtask.assigned_model = current_model
                        subtask.generated_text = gen_res.generated_text
                        subtask.actual_provider = gen_res.provider or curr_provider
                        subtask.provider = subtask.actual_provider
                        subtask.latency_ms = att_latency_ms
                        subtask.cost = getattr(gen_res, "cost", 0.0)
                        subtask.cost_currency = getattr(gen_res, "cost_currency", "USD")
                        subtask.cost_source = getattr(gen_res, "cost_source", "zero_local")
                        usage = getattr(gen_res, "usage", None)
                        if usage is not None:
                            subtask.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                            subtask.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                            subtask.total_tokens = int(getattr(usage, "total_tokens", 0) or (subtask.input_tokens + subtask.output_tokens))
                        subtask.validation_status = "passed"
                        if attempt > 1 or current_model != subtask.initial_model:
                            subtask.failover_used = True

                        subtask.attempts_detail.append({
                            "attempt_number": attempt,
                            "model": current_model,
                            "provider": subtask.provider,
                            "status": "success",
                            "validation_status": "passed",
                            "relevance_score": rel_score,
                            "error": None,
                            "latency_ms": att_latency_ms,
                            "timestamp": time.time()
                        })

                        if event_callback:
                            await self._emit_async(event_callback, {
                                "stage": "complex_execution",
                                "event_type": "provider_execution_completed",
                                "status": "completed",
                                "task_id": subtask.task_id,
                                "message": f"Subtask {subtask.task_id} completed successfully on {current_model} in {att_latency_ms:.1f}ms.",
                                "metadata": {
                                    "task_id": subtask.task_id,
                                    "model": current_model,
                                    "provider": subtask.provider,
                                    "failover_used": subtask.failover_used,
                                    "attempts": attempt,
                                    "latency_ms": att_latency_ms,
                                    "relevance_score": rel_score,
                                    "prompt_hash": subtask.prompt_hash
                                }
                            })
                        return subtask
                    else:
                        # DOMAIN RELEVANCE VALIDATION FAILURE
                        subtask.validation_status = "failed"
                        err_msg = f"[RELEVANCE VALIDATION FAILED] {rel_reason}"
                        subtask.attempts_detail.append({
                            "attempt_number": attempt,
                            "model": current_model,
                            "provider": curr_provider,
                            "status": "failed",
                            "validation_status": "failed",
                            "relevance_score": rel_score,
                            "failure_type": "invalid_response_domain",
                            "error": err_msg,
                            "latency_ms": att_latency_ms,
                            "timestamp": time.time()
                        })
                        attempted_models.add(current_model)

                        if event_callback:
                            await self._emit_async(event_callback, {
                                "stage": "complex_execution",
                                "event_type": "provider_attempt_failed",
                                "status": "attempt_failed",
                                "task_id": subtask.task_id,
                                "message": f"Attempt {attempt} failed domain check on {current_model}: {rel_reason}",
                                "metadata": {
                                    "task_id": subtask.task_id,
                                    "attempt": attempt,
                                    "model": current_model,
                                    "failure_type": "invalid_response_domain",
                                    "error": err_msg
                                }
                            })

                        # Select next candidate model to retry with EXACT SAME task objective / prompt hash
                        next_model = self._select_next_candidate(
                            subtask_description=subtask.task_objective,
                            category=subtask.category,
                            execution_mode=execution_mode,
                            attempted_models=attempted_models,
                            provider_tracker=provider_tracker
                        )
                        if next_model and next_model not in attempted_models:
                            current_model = next_model
                            continue
                        else:
                            subtask.error_message = err_msg
                            break

                else:
                    # Model Execution Provider Failure (429, timeout, server error)
                    err_msg = gen_res.error_message or "Execution returned empty response"
                    is_retryable, failure_type = classify_failure(err_msg)

                    subtask.attempts_detail.append({
                        "attempt_number": attempt,
                        "model": current_model,
                        "provider": curr_provider,
                        "status": "failed",
                        "failure_type": failure_type,
                        "error": err_msg,
                        "latency_ms": att_latency_ms,
                        "timestamp": time.time()
                    })

                    attempted_models.add(current_model)

                    if failure_type in ["quota_exhausted", "rate_limit"] and provider_tracker:
                        provider_tracker.mark_provider_exhausted(curr_provider, failure_type)

                    if event_callback:
                        await self._emit_async(event_callback, {
                            "stage": "complex_execution",
                            "event_type": "provider_attempt_failed",
                            "status": "attempt_failed",
                            "task_id": subtask.task_id,
                            "message": f"Attempt {attempt} failed on {current_model} ({curr_provider}): {err_msg}",
                            "metadata": {
                                "task_id": subtask.task_id,
                                "attempt": attempt,
                                "model": current_model,
                                "provider": curr_provider,
                                "failure_type": failure_type,
                                "error": err_msg
                            }
                        })

                    if not is_retryable:
                        logger.warning(f"[FAILOVER] Non-retryable error for {subtask.task_id} on {current_model}: {err_msg}")
                        subtask.error_message = err_msg
                        break

                    next_model = self._select_next_candidate(
                        subtask_description=subtask.task_objective,
                        category=subtask.category,
                        execution_mode=execution_mode,
                        attempted_models=attempted_models,
                        provider_tracker=provider_tracker
                    )

                    if next_model and next_model not in attempted_models:
                        if event_callback:
                            await self._emit_async(event_callback, {
                                "stage": "complex_execution",
                                "event_type": "provider_failover",
                                "status": "failover",
                                "task_id": subtask.task_id,
                                "message": f"[FAILOVER] {subtask.task_id}: {current_model} failed ({failure_type}); falling back to {next_model}",
                                "metadata": {
                                    "task_id": subtask.task_id,
                                    "primary_model": current_model,
                                    "failure_reason": err_msg,
                                    "failure_type": failure_type,
                                    "fallback_model": next_model,
                                    "attempts": attempt,
                                    "prompt_hash": subtask.prompt_hash
                                }
                            })
                        current_model = next_model
                    else:
                        subtask.error_message = f"All candidate models failed. Last error on {current_model}: {err_msg}"
                        break

            except (asyncio.TimeoutError, Exception) as exc:
                is_timeout = isinstance(exc, asyncio.TimeoutError)
                err_msg = "Local Ollama generation timed out (60s limit exceeded)" if is_timeout else str(exc)
                fail_type = "timeout" if is_timeout else "exception"
                t_attempt_end = time.perf_counter()
                att_latency_ms = round((t_attempt_end - t_attempt_start) * 1000, 2)
                attempted_models.add(current_model)

                logger.warning(f"[ATTEMPT FAILED] task_id={subtask.task_id} | model={current_model} | error={err_msg}")

                subtask.attempts_detail.append({
                    "attempt_number": attempt,
                    "model": current_model,
                    "provider": curr_provider,
                    "status": "failed",
                    "failure_type": fail_type,
                    "error": err_msg,
                    "latency_ms": att_latency_ms,
                    "timestamp": time.time()
                })

                next_model = self._select_next_candidate(
                    subtask_description=subtask.task_objective,
                    category=subtask.category,
                    execution_mode=execution_mode,
                    attempted_models=attempted_models,
                    provider_tracker=provider_tracker
                )
                if next_model and next_model not in attempted_models:
                    current_model = next_model
                else:
                    subtask.error_message = err_msg
                    break

        # If loop finished without success
        subtask.status = "FAILED"
        subtask.execution_success = False
        if not subtask.error_message:
            subtask.error_message = "All eligible models failed during provider failover."

        if event_callback:
            await self._emit_async(event_callback, {
                "stage": "complex_execution",
                "event_type": "task_failed",
                "status": "failed",
                "task_id": subtask.task_id,
                "message": f"Subtask {subtask.task_id} failed after {attempt} attempts: {subtask.error_message}",
                "metadata": {
                    "task_id": subtask.task_id,
                    "initial_model": subtask.initial_model,
                    "attempts": attempt,
                    "error": subtask.error_message,
                    "prompt_hash": subtask.prompt_hash
                }
            })

        return subtask

    def _select_next_candidate(
        self,
        subtask_description: str,
        category: str,
        execution_mode: str,
        attempted_models: Set[str],
        provider_tracker: Optional[RequestProviderTracker] = None
    ) -> Optional[str]:
        exec_mode = "online" if execution_mode.lower() in ["online", "mistral", "gemini"] else "local"
        ex_providers = provider_tracker.get_exhausted_list() if provider_tracker else []

        dec_req = DecisionRequest(
            text=f"Task: {subtask_description}\nCategory: {category}",
            execution_mode=exec_mode,
            excluded_models=list(attempted_models),
            excluded_providers=ex_providers
        )
        dec_res = self.decision_engine.decide(dec_req)

        sel_model = dec_res.selected_model
        if sel_model and sel_model not in attempted_models and sel_model != "BAAI/bge-m3":
            if exec_mode == "local":
                model_meta = self.decision_engine.model_registry.get_model(sel_model)
                if model_meta and model_meta.provider and "ollama" not in model_meta.provider.lower():
                    # Fast local Ollama candidates
                    all_models = self.decision_engine.model_registry.list_models()
                    local_cands = [
                        m.model_id for m in all_models 
                        if m.provider and "ollama" in m.provider.lower() 
                        and m.model_id not in attempted_models 
                        and m.model_id != "BAAI/bge-m3"
                    ]
                    # Sort to prefer fast 3B/4B models over heavy 7B models
                    local_cands.sort(key=lambda m: 0 if "gemma" in m or "qwen" in m else 1)
                    return local_cands[0] if local_cands else None
            return sel_model
        return None

    async def _emit_async(self, callback: Callable[[Dict[str, Any]], None], data: Dict[str, Any]):
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.warning(f"Error in SSE event_callback: {str(e)}")
