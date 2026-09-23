import time
import logging
from typing import List, Dict, Any, Optional, Callable

from app.schemas.complex import SubTask, ComplexTaskPlan
from app.services.complex.dynamic_decomposer import DynamicTaskDecomposer
from app.services.complex.dependency_graph import DependencyGraphBuilder
from app.services.complex.parallel_scheduler import ParallelTaskScheduler
from app.services.complex.task_aggregator import TaskAggregator

logger = logging.getLogger("orchestrator")

class ComplexTaskDecomposer:
    """
    Master Complex Task Orchestrator & Decomposer.
    - Determines if prompt requires multi-task decomposition.
    - Dynamically parses subtasks and DAG dependencies.
    - Routes subtasks through BaselineAdaptivePolicy (Production) and RLContextualBanditPolicy (Shadow Mode).
    - Concurrently executes independent subtasks via ParallelTaskScheduler.
    - Aggregates subtask outputs via TaskAggregator into a coherent final response.
    """

    def __init__(
        self,
        dynamic_decomposer: Optional[DynamicTaskDecomposer] = None,
        graph_builder: Optional[DependencyGraphBuilder] = None,
        parallel_scheduler: Optional[ParallelTaskScheduler] = None,
        task_aggregator: Optional[TaskAggregator] = None
    ):
        self.dynamic_decomposer = dynamic_decomposer or DynamicTaskDecomposer()
        self.graph_builder = graph_builder or DependencyGraphBuilder()
        self.parallel_scheduler = parallel_scheduler or ParallelTaskScheduler()
        self.task_aggregator = task_aggregator or TaskAggregator()

    def is_complex_prompt(self, prompt: str, complexity_score: float = 0.0, complexity_level: str = "") -> bool:
        return self.dynamic_decomposer.is_decomposable_complex(prompt, complexity_score, complexity_level)

    def decompose(
        self,
        prompt: str,
        complexity_score: float = 0.85,
        complexity_level: str = "very_high",
        execution_mode: str = "local",
        primary_selected_model: Optional[str] = None
    ) -> ComplexTaskPlan:
        """
        Generates structured task plan without running backend task execution.
        """
        t0 = time.perf_counter()
        if not prompt or not prompt.strip():
            raise ValueError("Prompt text for complex task decomposition cannot be empty.")

        prompt_clean = prompt.strip()

        # Step 1: Dynamic Subtask Decomposition & Model Allocation
        subtasks = self.dynamic_decomposer.decompose(prompt_clean, execution_mode=execution_mode)

        # Step 2: Compute Execution Levels via DAG Graph Builder
        execution_levels = self.graph_builder.compute_execution_levels(subtasks)

        t1 = time.perf_counter()
        plan_latency_ms = round((t1 - t0) * 1000, 2)

        return ComplexTaskPlan(
            is_complex=True,
            original_prompt=prompt_clean,
            subtasks=subtasks,
            execution_levels=execution_levels,
            total_subtasks=len(subtasks),
            plan_latency_ms=plan_latency_ms,
            aggregated_response=None,
            execution_success=False,
            total_execution_latency_ms=plan_latency_ms
        )

    async def execute_complex_plan_async(
        self,
        prompt: str,
        execution_mode: str = "local",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> ComplexTaskPlan:
        """
        Executes dynamic decomposition, parallel async DAG scheduling, and result aggregation.
        """
        t0 = time.perf_counter()
        logger.info(f"[COMPLEX START] prompt='{prompt[:60]}...' | execution_mode={execution_mode}")

        logger.info(f"[DECOMPOSITION START] Parsing prompt into dynamic subtasks...")
        plan = self.decompose(prompt, execution_mode=execution_mode)
        logger.info(f"[DECOMPOSITION COMPLETE] total_subtasks={plan.total_subtasks} | execution_levels={plan.execution_levels}")

        if event_callback:
            await self._emit_async(event_callback, {
                "stage": "complex_decomposition",
                "status": "completed",
                "message": f"Decomposed query into {plan.total_subtasks} subtasks across {len(plan.execution_levels)} execution levels.",
                "metadata": {
                    "total_subtasks": plan.total_subtasks,
                    "execution_levels": plan.execution_levels,
                    "subtasks": [st.dict() for st in plan.subtasks]
                }
            })

        # Execute Parallel Subtasks via DAG Scheduler
        from app.services.provider_failover import RequestProviderTracker
        provider_tracker = RequestProviderTracker()

        executed_subtasks = await self.parallel_scheduler.execute_plan_async(
            subtasks=plan.subtasks,
            execution_levels=plan.execution_levels,
            execution_mode=execution_mode,
            event_callback=event_callback,
            provider_tracker=provider_tracker
        )

        plan.subtasks = executed_subtasks

        # Check execution success across subtasks
        completed_tasks = [t for t in executed_subtasks if t.execution_success]
        plan.execution_success = len(completed_tasks) > 0

        # Aggregate Results
        logger.info(f"[AGGREGATION START] completed_subtasks={len(completed_tasks)}/{len(executed_subtasks)}")
        if event_callback:
            await self._emit_async(event_callback, {
                "stage": "complex_aggregation",
                "status": "running",
                "message": f"Aggregating {len(completed_tasks)}/{len(executed_subtasks)} completed subtasks into final response..."
            })

        aggregated_text = self.task_aggregator.aggregate_results(
            original_prompt=prompt,
            subtasks=executed_subtasks,
            execution_mode=execution_mode
        )

        plan.aggregated_response = aggregated_text

        t1 = time.perf_counter()
        plan.total_execution_latency_ms = round((t1 - t0) * 1000, 2)

        subtask_costs = [float(getattr(st, "cost", 0.0) or 0.0) for st in executed_subtasks]
        synthesis_cost = float(getattr(self.task_aggregator, "last_synthesis_cost", 0.0) or 0.0)
        plan.synthesis_cost = synthesis_cost
        plan.total_workflow_cost = round(sum(subtask_costs) + synthesis_cost, 6)
        plan.cost_currency = "USD"

        if plan.total_workflow_cost > 0:
            plan.cost_source = "configured_pricing_estimate" if any(float(getattr(st, "cost", 0.0) or 0.0) > 0 for st in executed_subtasks) or synthesis_cost > 0 else "unknown"
        else:
            plan.cost_source = "zero_local"

        plan.total_input_tokens = sum(int(getattr(st, "input_tokens", 0) or 0) for st in executed_subtasks)
        plan.total_output_tokens = sum(int(getattr(st, "output_tokens", 0) or 0) for st in executed_subtasks)
        plan.total_tokens = sum(int(getattr(st, "total_tokens", 0) or 0) for st in executed_subtasks)

        if getattr(self.task_aggregator, "last_synthesis_usage", None):
            usage = self.task_aggregator.last_synthesis_usage
            plan.total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            plan.total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            plan.total_tokens += int(getattr(usage, "total_tokens", 0) or (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0))

        successful_subtasks = [st for st in executed_subtasks if st.execution_success]
        plan.actual_models = list(dict.fromkeys(str(st.actual_model or st.assigned_model) for st in successful_subtasks if st.actual_model or st.assigned_model))
        plan.actual_providers = list(dict.fromkeys(str(st.actual_provider or st.provider) for st in successful_subtasks if st.actual_provider or st.provider))
        plan.successful_models = list(plan.actual_models)
        plan.successful_providers = list(plan.actual_providers)
        plan.final_model = plan.actual_models[0] if len(plan.actual_models) == 1 else ("Multiple Models" if plan.actual_models else None)
        plan.final_provider = plan.actual_providers[0] if len(plan.actual_providers) == 1 else ("Multiple Providers" if plan.actual_providers else None)
        plan.failover_used = any(bool(st.failover_used) for st in executed_subtasks)
        plan.local_cost = round(sum(float(st.cost or 0.0) for st in successful_subtasks if (st.actual_provider or st.provider) and "ollama" in (st.actual_provider or st.provider).lower()), 6)
        plan.cloud_cost = round(plan.total_workflow_cost - plan.local_cost, 6)
        plan.local_calls = sum(1 for st in successful_subtasks if (st.actual_provider or st.provider) and "ollama" in (st.actual_provider or st.provider).lower())
        plan.cloud_calls = len(successful_subtasks) - plan.local_calls
        plan.billable_calls = sum(1 for st in successful_subtasks if float(st.cost or 0.0) > 0.0) + (1 if synthesis_cost > 0.0 else 0)
        plan.calls = [
            {
                "task_id": st.task_id,
                "initial_model": st.initial_model or st.assigned_model,
                "model": st.actual_model or st.assigned_model,
                "provider": st.actual_provider or st.provider,
                "cost_usd": float(st.cost or 0.0),
                "cost_source": st.cost_source,
                "input_tokens": int(st.input_tokens or 0),
                "output_tokens": int(st.output_tokens or 0),
                "total_tokens": int(st.total_tokens or 0),
                "is_local": bool((st.actual_provider or st.provider) and "ollama" in (st.actual_provider or st.provider).lower()),
                "failover_used": bool(st.failover_used),
            }
            for st in successful_subtasks
        ]
        if synthesis_cost > 0.0 or getattr(self.task_aggregator, "last_synthesis_usage", None):
            synthesis_usage = getattr(self.task_aggregator, "last_synthesis_usage", None)
            plan.calls.append({
                "task_id": "SYNTHESIS",
                "model": None,
                "provider": "local_deterministic" if synthesis_cost == 0.0 else "LLM synthesis",
                "cost_usd": synthesis_cost,
                "cost_source": getattr(self.task_aggregator, "last_synthesis_source", "unknown"),
                "input_tokens": int(getattr(synthesis_usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(synthesis_usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(synthesis_usage, "total_tokens", 0) or 0),
                "is_local": synthesis_cost == 0.0,
                "failover_used": False,
            })

        logger.info("[EXECUTION_SUMMARY] %s", {
            "actual_models": plan.actual_models,
            "actual_providers": plan.actual_providers,
            "failover_used": plan.failover_used,
            "baseline_fallback": plan.baseline_fallback,
            "total_subtasks": plan.total_subtasks,
            "completed_subtasks": len(completed_tasks),
            "local_calls": plan.local_calls,
            "cloud_calls": plan.cloud_calls,
            "input_tokens": plan.total_input_tokens,
            "output_tokens": plan.total_output_tokens,
            "total_tokens": plan.total_tokens,
            "local_cost_usd": plan.local_cost,
            "cloud_cost_usd": plan.cloud_cost,
            "total_cost_usd": plan.total_workflow_cost,
            "cost_source": plan.cost_source,
        })

        logger.info(
            f"[AGGREGATION COMPLETE] total_execution_latency_ms={plan.total_execution_latency_ms:.1f}ms | "
            f"total_workflow_cost=${plan.total_workflow_cost:.6f}"
        )

        if event_callback:
            await self._emit_async(event_callback, {
                "stage": "complex_aggregation",
                "status": "completed",
                "message": f"Aggregation complete. Complex task finished in {plan.total_execution_latency_ms:.1f}ms.",
                "metadata": {
                    "total_execution_latency_ms": plan.total_execution_latency_ms,
                    "completed_subtasks": len(completed_tasks)
                }
            })

        return plan

    async def _emit_async(self, callback: Callable[[Dict[str, Any]], None], data: Dict[str, Any]):
        try:
            import asyncio
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.warning(f"Error in SSE event_callback: {str(e)}")
