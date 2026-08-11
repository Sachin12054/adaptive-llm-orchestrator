from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.schemas.decision import DecisionResponse
from app.schemas.response import ResponseGenerationResponse
from app.schemas.verification import VerificationResponse
from app.schemas.reward import RewardComputeResponse

class OrchestrationRequest(BaseModel):
    prompt: str = Field(..., description="User prompt text to orchestrate through the full platform pipeline")
    system_instruction: Optional[str] = Field(None, description="Optional system instruction / persona prompt")
    temperature: Optional[float] = Field(None, description="Optional sampling temperature [0.0 - 2.0]")
    max_output_tokens: Optional[int] = Field(None, description="Optional maximum output token limit")
    execution_mode: Optional[str] = Field("local", description="Execution mode: local (Ollama) or gemini (Google Gemini API)")
    run_id: Optional[str] = Field(None, description="Unique execution run identifier for request-response correlation")

class OrchestrationResponse(BaseModel):
    run_id: Optional[str] = Field(None, description="Unique execution run identifier")
    success: bool = Field(..., description="True if end-to-end orchestration pipeline executed successfully")
    prompt: str = Field(..., description="Original user prompt text")
    selected_model: Optional[str] = Field(None, description="Model identifier selected by Step 15 Adaptive Decision Engine")
    decision_score: float = Field(0.0, description="Winning model decision score")
    decision: DecisionResponse = Field(..., description="Full Step 15 Decision Engine response payload")
    generation: ResponseGenerationResponse = Field(..., description="Full Step 16 Response Generator payload")
    verification: VerificationResponse = Field(..., description="Full Step 17 Response Verifier payload")
    reward: RewardComputeResponse = Field(..., description="Full Step 18 Reward Signal payload")
    pipeline_latency_ms: float = Field(..., description="End-to-end orchestration pipeline latency in milliseconds")

class OrchestrationStatusResponse(BaseModel):
    status: str = Field(..., description="E2E Pipeline status: ready, unconfigured")
    service: str = Field("e2e_orchestrator", description="Service identifier")
    gemini_configured: bool = Field(..., description="True if GEMINI_API_KEY environment variable is configured")
    active_policy: str = Field("baseline_adaptive_policy", description="Active decision policy identifier")
