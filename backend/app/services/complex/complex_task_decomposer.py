import time
import re
import logging
from typing import List, Dict, Any, Optional

from app.schemas.complex import SubTask, ComplexTaskPlan
from app.services.complex.task_allocator import TaskAllocator
from app.services.complex.dependency_graph import DependencyGraphBuilder

logger = logging.getLogger("orchestrator")

class ComplexTaskDecomposer:
    """
    Decomposes complex user prompts into structured subtasks,
    assigns model candidates via TaskAllocator, and computes parallel execution levels.
    PLAN GENERATION ONLY - Subtasks are not executed in this milestone.
    """

    def __init__(
        self,
        allocator: Optional[TaskAllocator] = None,
        graph_builder: Optional[DependencyGraphBuilder] = None
    ):
        self.allocator = allocator or TaskAllocator()
        self.graph_builder = graph_builder or DependencyGraphBuilder()

    def is_complex_prompt(self, prompt: str, complexity_score: float = 0.0, complexity_level: str = "") -> bool:
        if complexity_level.lower() in ["very_high", "complex"] or complexity_score >= 0.80:
            return True

        # Check multi-step indicator keywords
        indicators = ["and", "then", "after", "calculate", "clean", "explain", "implement", "build", "scrape", "design"]
        matched = [kw for kw in indicators if re.search(rf"\b{kw}\b", prompt.lower())]
        return len(matched) >= 3 or len(prompt.split(",")) >= 3

    def decompose(
        self,
        prompt: str,
        complexity_score: float = 0.85,
        complexity_level: str = "very_high",
        execution_mode: str = "local",
        primary_selected_model: Optional[str] = None
    ) -> ComplexTaskPlan:
        t0 = time.perf_counter()

        if not prompt or not prompt.strip():
            raise ValueError("Prompt text for complex task decomposition cannot be empty.")

        prompt_clean = prompt.strip()

        # Step 1: Query BaselineAdaptivePolicy if primary_selected_model is missing
        if not primary_selected_model or not primary_selected_model.strip():
            try:
                from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
                from app.schemas.decision import DecisionRequest
                engine = AdaptiveDecisionEngine()
                dec_res = engine.decide(DecisionRequest(text=prompt_clean, execution_mode=execution_mode))
                primary_selected_model = dec_res.selected_model
            except Exception as dec_err:
                logger.error(f"Failed to query BaselineAdaptivePolicy for complex task decomposition: {str(dec_err)}")

        # Step 2: Subtask Generation & Category Assignment
        subtasks_data = self._generate_subtasks(prompt_clean)

        # Step 3: Assign Model for each Subtask via Authoritative Policy Decision
        subtasks: List[SubTask] = []
        for item in subtasks_data:
            assigned_model = self.allocator.allocate_model(
                item["category"],
                execution_mode=execution_mode,
                policy_selected_model=primary_selected_model
            )
            subtasks.append(SubTask(
                task_id=item["task_id"],
                description=item["description"],
                category=item["category"],
                assigned_model=assigned_model,
                dependencies=item.get("dependencies", [])
            ))

        # Step 3: Compute Execution Levels via DAG
        execution_levels = self.graph_builder.compute_execution_levels(subtasks)

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 2)

        logger.info(f"ComplexTaskDecomposer generated plan with {len(subtasks)} subtasks across {len(execution_levels)} execution levels in {latency_ms} ms.")

        return ComplexTaskPlan(
            is_complex=True,
            original_prompt=prompt_clean,
            subtasks=subtasks,
            execution_levels=execution_levels,
            total_subtasks=len(subtasks),
            plan_latency_ms=latency_ms
        )

    def _generate_subtasks(self, prompt: str) -> List[Dict[str, Any]]:
        prompt_lower = prompt.lower()

        # Check for scraper / web app workflow pattern
        if "scraper" in prompt_lower or "scrape" in prompt_lower:
            return [
                {
                    "task_id": "TASK-1",
                    "description": "Understand the required scraping workflow and identify required data fields.",
                    "category": "general",
                    "dependencies": []
                },
                {
                    "task_id": "TASK-2",
                    "description": "Design and implement the web scraper script.",
                    "category": "coding",
                    "dependencies": ["TASK-1"]
                },
                {
                    "task_id": "TASK-3",
                    "description": "Design data cleaning, normalization, and validation logic.",
                    "category": "reasoning",
                    "dependencies": ["TASK-1"]
                },
                {
                    "task_id": "TASK-4",
                    "description": "Calculate statistical metrics and aggregates from the cleaned dataset.",
                    "category": "mathematics",
                    "dependencies": ["TASK-3"]
                },
                {
                    "task_id": "TASK-5",
                    "description": "Synthesize results and explain the final implementation and insights.",
                    "category": "explanation",
                    "dependencies": ["TASK-2", "TASK-4"]
                }
            ]

        # Check for general coding + analysis workflow pattern
        if "build" in prompt_lower or "implement" in prompt_lower or "code" in prompt_lower:
            return [
                {
                    "task_id": "TASK-1",
                    "description": "Analyze system requirements and define module architecture.",
                    "category": "general",
                    "dependencies": []
                },
                {
                    "task_id": "TASK-2",
                    "description": "Implement core code modules and functions.",
                    "category": "coding",
                    "dependencies": ["TASK-1"]
                },
                {
                    "task_id": "TASK-3",
                    "description": "Analyze algorithm trade-offs and logical performance constraints.",
                    "category": "reasoning",
                    "dependencies": ["TASK-1"]
                },
                {
                    "task_id": "TASK-4",
                    "description": "Document code implementation, API usage, and operational guidelines.",
                    "category": "explanation",
                    "dependencies": ["TASK-2", "TASK-3"]
                }
            ]

        # Generic multi-step analytical fallback decomposition
        return [
            {
                "task_id": "TASK-1",
                "description": "Deconstruct problem statement and establish analytical baseline.",
                "category": "general",
                "dependencies": []
            },
            {
                "task_id": "TASK-2",
                "description": "Formulate logical reasoning model and evaluate constraints.",
                "category": "reasoning",
                "dependencies": ["TASK-1"]
            },
            {
                "task_id": "TASK-3",
                "description": "Generate technical implementation and analytical summary.",
                "category": "explanation",
                "dependencies": ["TASK-2"]
            }
        ]
