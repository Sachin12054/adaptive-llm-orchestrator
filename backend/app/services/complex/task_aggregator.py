import time
import logging
from typing import List, Optional

from app.schemas.complex import SubTask
from app.services.response_generator import ResponseGenerator
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.complex.synthesis_strategies import (
    deterministic_s2_assembly,
    hybrid_s3_assembly,
    llm_synthesis_s0_s1
)

logger = logging.getLogger("orchestrator")

class TaskAggregator:
    """
    Task Result Aggregator & Synthesizer.
    - Default synthesis mode: S2 (Deterministic Structured Assembly - 0ms LLM overhead).
    - Configurable modes: S0 (DeepSeek 7B), S1 (Gemma 4B), S3 (Hybrid Assembly + Summary).
    - Gracefully handles partial completion / failed subtasks without crashing.
    """

    def __init__(
        self,
        response_generator: Optional[ResponseGenerator] = None,
        decision_engine: Optional[AdaptiveDecisionEngine] = None
    ):
        self.response_generator = response_generator or ResponseGenerator()
        self.decision_engine = decision_engine or AdaptiveDecisionEngine()

    def aggregate_results(
        self,
        original_prompt: str,
        subtasks: List[SubTask],
        execution_mode: str = "local",
        synthesis_strategy: str = "S2"
    ) -> str:
        """
        Combines completed subtask outputs into a coherent final response.
        By default, uses S2 Deterministic Structured Assembly (0 LLM overhead).
        """
        self.last_synthesis_cost = 0.0
        self.last_synthesis_source = "zero_local"
        self.last_synthesis_usage = None

        completed = [t for t in subtasks if t.execution_success and t.generated_text]

        if not completed:
            return f"Complex task execution failed across all {len(subtasks)} subtasks. Error: No subtask outputs available."

        # Single subtask direct return
        if len(subtasks) == 1:
            return completed[0].generated_text

        strategy = synthesis_strategy.upper().strip()

        if strategy == "S2":
            text, latency_ms = deterministic_s2_assembly(original_prompt, subtasks)
            self.last_synthesis_cost = 0.0
            self.last_synthesis_source = "zero_local"
            self.last_synthesis_usage = None
            logger.info(f"[TASK AGGREGATOR] S2 Deterministic Assembly complete in {latency_ms} ms (0 LLM calls).")
            return text
        elif strategy == "S3":
            text, asm_ms, sum_ms, fallback, synthesis_response = hybrid_s3_assembly(original_prompt, subtasks, self.response_generator, execution_mode)
            self.last_synthesis_cost = float(getattr(synthesis_response, "cost", 0.0) or 0.0)
            self.last_synthesis_source = getattr(synthesis_response, "cost_source", "zero_local") if synthesis_response else "zero_local"
            self.last_synthesis_usage = getattr(synthesis_response, "usage", None)
            logger.info(f"[TASK AGGREGATOR] S3 Hybrid Assembly complete (asm={asm_ms}ms, sum={sum_ms}ms, fallback={fallback}).")
            return text
        elif strategy == "S1":
            text, synth_ms, synthesis_response = llm_synthesis_s0_s1(original_prompt, subtasks, "gemma-3-4b", self.response_generator, execution_mode)
            self.last_synthesis_cost = float(getattr(synthesis_response, "cost", 0.0) or 0.0)
            self.last_synthesis_source = getattr(synthesis_response, "cost_source", "zero_local") if synthesis_response else "zero_local"
            self.last_synthesis_usage = getattr(synthesis_response, "usage", None)
            logger.info(f"[TASK AGGREGATOR] S1 Gemma 4B Synthesis complete in {synth_ms} ms.")
            return text
        elif strategy == "S0":
            text, synth_ms, synthesis_response = llm_synthesis_s0_s1(original_prompt, subtasks, "deepseek-r1-7b", self.response_generator, execution_mode)
            self.last_synthesis_cost = float(getattr(synthesis_response, "cost", 0.0) or 0.0)
            self.last_synthesis_source = getattr(synthesis_response, "cost_source", "zero_local") if synthesis_response else "zero_local"
            self.last_synthesis_usage = getattr(synthesis_response, "usage", None)
            logger.info(f"[TASK AGGREGATOR] S0 DeepSeek 7B Synthesis complete in {synth_ms} ms.")
            return text
        else:
            # Default fallback to S2
            text, latency_ms = deterministic_s2_assembly(original_prompt, subtasks)
            return text
