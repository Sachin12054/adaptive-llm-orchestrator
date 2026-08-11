from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SubTask(BaseModel):
    task_id: str = Field(..., description="Unique subtask identifier, e.g. TASK-1")
    description: str = Field(..., description="Detailed description of subtask objective")
    category: str = Field(..., description="Subtask category: general, coding, reasoning, mathematics, explanation, factual, summarization")
    assigned_model: str = Field(..., description="Assigned local Ollama model ID: gemma-3-4b, qwen-coder-3b, deepseek-r1-7b")
    dependencies: List[str] = Field(default_factory=list, description="List of prerequisite subtask IDs")

class ComplexTaskPlan(BaseModel):
    is_complex: bool = Field(True, description="True if prompt was classified as a complex task requiring decomposition")
    original_prompt: str = Field(..., description="Original complex user prompt")
    subtasks: List[SubTask] = Field(..., description="List of decomposed subtasks with model allocations and dependencies")
    execution_levels: List[List[str]] = Field(..., description="Ordered parallel execution groups (levels of independent subtasks)")
    total_subtasks: int = Field(..., description="Total count of decomposed subtasks")
    plan_latency_ms: float = Field(..., description="Decomposition and task plan generation latency in milliseconds")

class ComplexPlanRequest(BaseModel):
    prompt: str = Field(..., description="Complex user prompt to decompose into a task allocation plan")
    execution_mode: Optional[str] = Field("local", description="Execution mode: local or online")
    primary_selected_model: Optional[str] = Field(None, description="Authoritative policy selected model from BaselineAdaptivePolicy")
