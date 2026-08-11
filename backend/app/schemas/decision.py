from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class IntentInput(BaseModel):
    intent: str
    top_similarity: float
    second_similarity: float
    margin: float
    is_ambiguous: bool

class ComplexityInput(BaseModel):
    level: str
    complexity_score: float
    semantic_complexity: float
    reasoning_complexity: float
    task_complexity: float
    context_complexity: float
    output_complexity: float

class ResourceInput(BaseModel):
    cpu_utilization_percent: float
    memory_utilization_percent: float
    gpu_available: bool
    available_memory_gb: float
    execution_device: str
    gpu_total_vram_gb: Optional[float] = None
    gpu_free_vram_gb: Optional[float] = None
    gpu_used_vram_gb: Optional[float] = None

class DecisionRequest(BaseModel):
    text: str = Field(..., description="User prompt text to analyze and make model routing decision for")
    intent: Optional[IntentInput] = Field(None, description="Optional pre-computed Step 9 intent analysis")
    complexity: Optional[ComplexityInput] = Field(None, description="Optional pre-computed Step 10 complexity analysis")
    resources: Optional[ResourceInput] = Field(None, description="Optional pre-computed Step 11 resource telemetry")
    execution_mode: Optional[str] = Field("local", description="Execution mode: local or online")
    excluded_models: Optional[List[str]] = Field(default_factory=list, description="Optional list of failed model IDs to exclude during fallback re-evaluation")

class CandidateScoreBreakdown(BaseModel):
    model_id: str = Field(..., description="Candidate model identifier")
    provider: Optional[str] = Field(None, description="Model provider or host (e.g. Local Ollama, Mistral API, Google Gemini API, Groq API, OpenRouter API)")
    display_name: Optional[str] = Field(None, description="Human-readable model name")
    eligible: bool = Field(..., description="True if candidate satisfies availability and capability prerequisites")
    ineligible_reason: Optional[str] = Field(None, description="Reason if model is ineligible for selection")
    capability_score: float = Field(0.0, description="Capability match score [0.0 - 1.0]")
    complexity_fit_score: float = Field(0.0, description="Complexity fit score [0.0 - 1.0]")
    resource_fit_score: float = Field(0.0, description="Resource state fit score [0.0 - 1.0]")
    context_fit_score: float = Field(0.0, description="Context length fit score [0.0 - 1.0]")
    candidate_score: float = Field(0.0, description="Transparent weighted candidate decision score [0.0 - 1.0]")

class DecisionTrace(BaseModel):
    decision_id: str = Field(..., description="Unique decision execution identifier")
    intent: str = Field(..., description="Primary intent classification")
    is_ambiguous: bool = Field(..., description="True if intent classification margin was below ambiguity threshold")
    complexity_level: str = Field(..., description="Calculated prompt complexity level")
    complexity_score: float = Field(..., description="Overall weighted prompt complexity score")
    resource_summary: Dict[str, Any] = Field(..., description="Summary of system resource state used for decision")
    selected_model: Optional[str] = Field(None, description="Model selected for execution")
    decision_score: float = Field(0.0, description="Winning candidate decision score")
    policy: str = Field("baseline_adaptive_policy", description="Decision policy identifier used")
    shadow_rl_decision: Optional[Dict[str, Any]] = Field(None, description="Shadow RL policy proposal and prediction score")
    decision_latency_ms: float = Field(..., description="Decision policy execution latency in milliseconds")

class DecisionResponse(BaseModel):
    text: str = Field(..., description="Analyzed prompt text")
    selected_model: Optional[str] = Field(None, description="Selected model identifier for Step 14 execution")
    decision_score: float = Field(0.0, description="Winning candidate score [0.0 - 1.0]")
    policy: str = Field("baseline_adaptive_policy", description="Decision policy name")
    reasoning: List[str] = Field(..., description="Bulleted explanatory justification for winning model selection")
    candidates: List[CandidateScoreBreakdown] = Field(..., description="Ranked list of evaluated candidate models")
    decision_trace: DecisionTrace = Field(..., description="Full structured trace for analytics and future RL logging")
    shadow_rl_decision: Optional[Dict[str, Any]] = Field(None, description="Shadow RL policy proposal and prediction score")
    intent_info: Optional[Dict[str, Any]] = Field(None, description="Detailed Step 9 intent analysis")
    complexity_info: Optional[Dict[str, Any]] = Field(None, description="Detailed Step 10 complexity analysis")
    total_pipeline_latency_ms: float = Field(..., description="Total decision engine pipeline latency in milliseconds")

class DecisionStatusResponse(BaseModel):
    status: str = Field(..., description="Engine status: ready, unconfigured, no_executable_models")
    policy: str = Field("baseline_adaptive_policy", description="Active policy identifier")
    registered_models_count: int = Field(..., description="Total models in registry")
    executable_candidates_count: int = Field(..., description="Number of currently executable LLM candidate models")
    executable_models: List[str] = Field(..., description="List of currently executable model IDs")
