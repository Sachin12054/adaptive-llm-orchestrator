import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.schemas.complex import SubTask
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.schemas.decision import DecisionRequest
from app.services.experience_buffer import ACTION_MAP, REVERSE_ACTION_MAP, ExperienceBufferService

logger = logging.getLogger("orchestrator")

# Action 7 (BAAI/bge-m3) is an embedding model and MUST NEVER be selected for normal LLM response generation
EMBEDDING_ACTION_INDEX = 7
EMBEDDING_MODEL_ID = "BAAI/bge-m3"

class DynamicTaskDecomposer:
    """
    Dynamic Task Decomposition and Task-Level Multi-LLM Model Router.
    - Dynamically decomposes complex multi-objective user prompts into structured subtasks with DAG dependencies.
    - Routes EACH subtask through BaselineAdaptivePolicy for authoritative production model selection.
    - Evaluates EACH subtask in SHADOW MODE using RLContextualBanditPolicy.
    - Strictly masks Action 7 (BAAI/bge-m3) from response generation.
    """

    def __init__(
        self,
        decision_engine: Optional[AdaptiveDecisionEngine] = None,
        rl_policy: Optional[RLContextualBanditPolicy] = None,
        experience_buffer: Optional[ExperienceBufferService] = None
    ):
        self.decision_engine = decision_engine or AdaptiveDecisionEngine()
        self.rl_policy = rl_policy or RLContextualBanditPolicy()
        self.exp_service = experience_buffer or ExperienceBufferService()

    def is_decomposable_complex(self, prompt: str, complexity_score: float = 0.0, complexity_level: str = "") -> bool:
        if not prompt or not prompt.strip():
            return False

        p_lower = prompt.lower().strip()
        comp_level_lower = str(complexity_level).lower().strip()

        # Always decompose VERY_HIGH complexity queries or explicitly multi-part queries
        if comp_level_lower in ["very_high", "complex"] or complexity_score >= 0.5564:
            return True

        # Check for multiple distinct question marks or comma-separated objectives
        question_count = prompt.count("?")
        if question_count >= 2:
            return True

        # Check for multi-clause conjunction indicators ("compare ... and recommend", "explain ... and implement", etc.)
        multi_task_indicators = ["compare", "recommend", "design", "explain", "implement", "calculate", "analyze", "evaluate", "list"]
        matched_kws = [kw for kw in multi_task_indicators if re.search(rf"\b{kw}\b", p_lower)]
        
        if len(matched_kws) >= 2 or len(prompt.split(",")) >= 3:
            return True

        return False

    def decompose(
        self,
        prompt: str,
        execution_mode: str = "local"
    ) -> List[SubTask]:
        """
        Dynamically analyzes the prompt text and generates a structured list of subtasks with DAG dependencies.
        Subtask count and structure are generated dynamically based on prompt contents.
        """
        prompt_clean = prompt.strip()
        subtask_specs = self._parse_prompt_subtasks(prompt_clean)

        from app.services.complex.task_response_validator import TaskResponseValidator

        subtasks: List[SubTask] = []
        for spec in subtask_specs:
            t_id = spec["task_id"]
            desc = spec["description"]
            cat = spec["category"]
            deps = spec.get("dependencies", [])

            # Compute immutable prompt hash with global prompt context
            p_hash = TaskResponseValidator.compute_prompt_hash(t_id, prompt_clean, desc)

            # Determine task complexity output token budget (tight budgets to prevent long generation hangs)
            if cat in ["coding", "reasoning", "mathematics"]:
                max_tokens = 750
            elif cat in ["explanation", "factual"]:
                max_tokens = 500
            else:
                max_tokens = 400

            # Perform feature-driven model routing for THIS specific subtask
            prod_model, rl_shadow_model, rl_agreed, q_val, provider = self.route_subtask(
                subtask_description=desc,
                category=cat,
                execution_mode=execution_mode,
                overall_prompt=prompt_clean
            )

            subtask_obj = SubTask(
                task_id=t_id,
                description=desc,
                category=cat,
                assigned_model=prod_model,
                dependencies=deps,
                status="PENDING",
                provider=provider,
                rl_shadow_model=rl_shadow_model,
                rl_agreement=rl_agreed,
                rl_predicted_q_value=q_val,
                execution_success=False,
                latency_ms=0.0,
                original_user_prompt=prompt_clean,
                task_objective=desc,
                prompt_hash=p_hash,
                max_output_tokens=max_tokens,
                validation_status="unverified"
            )
            subtasks.append(subtask_obj)

        return subtasks

    def route_subtask(
        self,
        subtask_description: str,
        category: str,
        execution_mode: str = "local",
        overall_prompt: str = ""
    ) -> Tuple[str, str, bool, float, str]:
        """
        Routes an individual subtask:
        1. BaselineAdaptivePolicy -> Authoritative Production Model Selection
        2. RLContextualBanditPolicy -> Shadow Recommendation + Q-value Estimation
        3. Enforces Action 7 (BAAI/bge-m3) Masking.
        """
        exec_mode = "online" if execution_mode.lower() in ["online", "mistral", "gemini"] else "local"

        # 1. Baseline Policy Decision
        dec_req = DecisionRequest(
            text=f"Task: {subtask_description}\nCategory: {category}",
            execution_mode=exec_mode
        )
        dec_res = self.decision_engine.decide(dec_req)
        prod_model = dec_res.selected_model or ("gemma-3-4b" if exec_mode == "local" else "gemini-3.5-flash")

        # Mask Action 7 (BAAI/bge-m3) from Production
        if prod_model == EMBEDDING_MODEL_ID:
            prod_model = "gemma-3-4b" if exec_mode == "local" else "gemini-3.5-flash"

        # Obtain Provider Metadata
        model_meta = self.decision_engine.model_registry.get_model(prod_model)
        provider = model_meta.provider if model_meta else ("Local Ollama" if exec_mode == "local" else "Online Cloud API")

        # 2. Construct Subtask Feature State Vector for RL Shadow Evaluation
        intent_info = {"intent": category, "is_ambiguous": False}
        resource_snapshot = self.decision_engine.resource_analyzer.get_resource_snapshot()

        # Compute fast subtask feature complexity without PyTorch model re-inference
        subtask_words = len(subtask_description.split())
        cmplx_score = round(min(1.0, max(0.1, subtask_words / 40.0)), 4)
        cmplx_level = "low" if cmplx_score < 0.33 else ("high" if cmplx_score > 0.66 else "medium")

        class FastSubtaskFactors:
            semantic_complexity = cmplx_score
            reasoning_complexity = cmplx_score
            task_complexity = cmplx_score
            context_complexity = cmplx_score
            output_complexity = cmplx_score
        
        class FastSubtaskCmplx:
            complexity_score = cmplx_score
            complexity_level = cmplx_level
            factors = FastSubtaskFactors()

        complexity_info = FastSubtaskCmplx()

        # Synthesize state vector for subtask
        dummy_trace = type("TraceMock", (), {
            "intent": category,
            "is_ambiguous": False,
            "complexity_score": complexity_info.complexity_score,
            "resource_summary": {
                "cpu_utilization_percent": resource_snapshot.cpu.utilization_percent,
                "memory_utilization_percent": resource_snapshot.memory.utilization_percent,
                "gpu_available": resource_snapshot.gpu.available
            }
        })()
        
        dummy_res = type("OrchestrationResponseMock", (), {
            "prompt": subtask_description,
            "decision_score": dec_res.decision_score,
            "decision": type("DecisionMock", (), {
                "decision_trace": dummy_trace,
                "complexity": complexity_info
            })()
        })()

        s_task = self.exp_service.state_encoder.encode_state(dummy_res)

        # 3. RL Shadow Decision Evaluation
        all_models = self.decision_engine.model_registry.list_models()
        valid_candidates = [m for m in all_models if m.model_id != EMBEDDING_MODEL_ID]

        shadow_dec = self.rl_policy.predict_shadow_decision(
            state_vector=s_task,
            baseline_selected_model=prod_model,
            candidate_models=valid_candidates
        )

        rl_shadow_model = shadow_dec.proposed_model
        if rl_shadow_model == EMBEDDING_MODEL_ID or shadow_dec.action_index == EMBEDDING_ACTION_INDEX:
            rl_shadow_model = prod_model

        rl_agreement = (prod_model == rl_shadow_model)
        q_value = float(shadow_dec.predicted_reward)

        logger.info(f"[TASK ROUTING] subtask='{subtask_description[:40]}...' | category={category} | production_model={prod_model} | rl_shadow_model={rl_shadow_model} | agreed={rl_agreement}")

        return prod_model, rl_shadow_model, rl_agreement, q_value, provider

    def _parse_prompt_subtasks(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Parses user prompt into subtask descriptions, categories, and dependencies dynamically.
        Supports question marks splitting, multi-clause conjunction splitting, and workflow patterns.
        """
        p_lower = prompt.lower()
        subtasks = []

        # Pattern 1: Question marks split (e.g. User example: "What is capital? What are places? What are hotels?...")
        if prompt.count("?") >= 2:
            parts = [p.strip() + "?" for p in prompt.split("?") if p.strip()]
            for idx, part in enumerate(parts, start=1):
                clean_part = part.rstrip("?").strip()
                cat = self._infer_category(clean_part)
                
                # Check if final part depends on previous parts (e.g., "compare and recommend", "summary")
                deps = []
                if idx > 1 and ("recommend" in clean_part.lower() or "compare" in clean_part.lower() or "summary" in clean_part.lower()):
                    deps = [f"TASK-{i}" for i in range(1, idx)]

                subtasks.append({
                    "task_id": f"TASK-{idx}",
                    "description": clean_part,
                    "category": cat,
                    "dependencies": deps
                })
            
            # If no synthesis task was created and we have >= 3 tasks, add a synthesis task
            if len(subtasks) >= 3 and not subtasks[-1]["dependencies"]:
                task_ids = [t["task_id"] for t in subtasks]
                subtasks.append({
                    "task_id": f"TASK-{len(subtasks)+1}",
                    "description": f"Synthesize and present a complete structured response for: {prompt[:80]}...",
                    "category": "explanation",
                    "dependencies": task_ids
                })
            return subtasks

        # Pattern 2: Multi-objective conjunction / comparison (e.g. "Compare Python and Java and recommend")
        if "compare" in p_lower or "versus" in p_lower or " vs " in p_lower:
            match = re.search(r"compare\s+([a-zA-Z0-9_\s]+)\s+(?:and|vs|versus)\s+([a-zA-Z0-9_\s]+)", p_lower)
            item1 = match.group(1).strip() if match else "first option"
            item2 = match.group(2).strip() if match else "second option"

            return [
                {
                    "task_id": "TASK-1",
                    "description": f"Analyze characteristics, strengths, and architecture of {item1}.",
                    "category": "reasoning",
                    "dependencies": []
                },
                {
                    "task_id": "TASK-2",
                    "description": f"Analyze characteristics, strengths, and architecture of {item2}.",
                    "category": "reasoning",
                    "dependencies": []
                },
                {
                    "task_id": "TASK-3",
                    "description": f"Compare trade-offs and performance between {item1} and {item2}.",
                    "category": "explanation",
                    "dependencies": ["TASK-1", "TASK-2"]
                },
                {
                    "task_id": "TASK-4",
                    "description": f"Provide final tailored recommendation based on user constraints.",
                    "category": "explanation",
                    "dependencies": ["TASK-3"]
                }
            ]

        # Pattern 3: Comprehensive Software Architecture / Multi-topic queries
        if len(prompt.split(",")) >= 3 or len(prompt.split()) >= 25:
            clauses = [c.strip() for c in prompt.split(",") if c.strip()]
            if len(clauses) >= 3:
                dep_tasks = []
                for idx, clause in enumerate(clauses[:6], start=1):
                    cat = self._infer_category(clause)
                    t_id = f"TASK-{idx}"
                    dep_tasks.append(t_id)
                    subtasks.append({
                        "task_id": t_id,
                        "description": clause,
                        "category": cat,
                        "dependencies": []
                    })
                
                # Final synthesis task
                subtasks.append({
                    "task_id": f"TASK-{len(subtasks)+1}",
                    "description": "Synthesize all analyzed components into a cohesive final architectural response.",
                    "category": "explanation",
                    "dependencies": dep_tasks
                })
                return subtasks

        # Default multi-step fallback
        return [
            {
                "task_id": "TASK-1",
                "description": f"Understand core objective and extract foundational requirements for: {prompt[:60]}",
                "category": "general",
                "dependencies": []
            },
            {
                "task_id": "TASK-2",
                "description": f"Execute in-depth technical analysis and reasoning.",
                "category": "reasoning",
                "dependencies": ["TASK-1"]
            },
            {
                "task_id": "TASK-3",
                "description": f"Synthesize final comprehensive response.",
                "category": "explanation",
                "dependencies": ["TASK-2"]
            }
        ]

    def _infer_category(self, text: str) -> str:
        t_lower = text.lower()
        if any(w in t_lower for w in ["code", "python", "javascript", "script", "function", "api", "html", "css", "bug", "implement"]):
            return "coding"
        if any(w in t_lower for w in ["calculate", "math", "equation", "sum", "multiply", "number", "formula"]):
            return "mathematics"
        if any(w in t_lower for w in ["why", "compare", "reason", "evaluate", "tradeoff", "architecture", "design"]):
            return "reasoning"
        if any(w in t_lower for w in ["explain", "describe", "how", "details", "overview"]):
            return "explanation"
        if any(w in t_lower for w in ["capital", "who", "what is", "where", "when"]):
            return "factual"
        return "general"
