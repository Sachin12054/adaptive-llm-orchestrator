from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class RewardComputeRequest(BaseModel):
    prompt: str = Field(..., description="Original user prompt text")
    selected_model: str = Field(..., description="Model identifier selected for execution")
    winning_score: Optional[float] = Field(None, description="Winning decision score from Step 15")
    execution_success: bool = Field(..., description="True if generation execution completed successfully")
    execution_status: str = Field(..., description="Execution status (e.g. completed, not_configured, failed)")
    generated_text: Optional[str] = Field(None, description="Generated response text")
    verification_status: str = Field(..., description="Verification status from Step 17 Response Verifier")
    verified: bool = Field(..., description="True if response passed baseline structural/relevance verification")
    response_present: bool = Field(..., description="True if non-empty text response is present")
    structural_quality_score: float = Field(0.0, description="Structural quality score [0.0 - 1.0]")
    completeness_score: float = Field(0.0, description="Response completeness score [0.0 - 1.0]")
    relevance_score: float = Field(0.0, description="Prompt-response relevance score [0.0 - 1.0]")
    factual_verification_status: str = Field("not_verified", description="Factual verification status")

class ComponentContribution(BaseModel):
    score: float = Field(..., description="Raw component metric score [0.0 - 1.0]")
    weight: float = Field(..., description="Component allocation weight")
    contribution: float = Field(..., description="Calculated contribution value (score * weight)")

class RewardBreakdown(BaseModel):
    quality: ComponentContribution = Field(..., description="Structural quality component (weight: 0.30)")
    completeness: ComponentContribution = Field(..., description="Completeness component (weight: 0.20)")
    relevance: ComponentContribution = Field(..., description="Prompt relevance component (weight: 0.25)")
    verification: ComponentContribution = Field(..., description="Verification baseline pass component (weight: 0.15)")
    execution: ComponentContribution = Field(..., description="Successful execution component (weight: 0.10)")

class RewardComputeResponse(BaseModel):
    success: bool = Field(..., description="True if reward signal computation completed successfully")
    selected_model: str = Field(..., description="Target model identifier evaluated")
    reward: float = Field(..., description="Normalized baseline reward signal [0.0 - 1.0]")
    reward_breakdown: RewardBreakdown = Field(..., description="Detailed component breakdown of calculated reward signal")
    reward_status: str = Field(..., description="Status: completed_verified, completed_unverified, failed_execution, not_verifiable")
    reasoning: List[str] = Field(..., description="Bulleted explanatory justification items for reward computation")
    factual_verification_status: str = Field("not_verified", description="Factual verification status")
    latency_ms: float = Field(..., description="Reward signal computation latency in milliseconds")

class RewardStatusResponse(BaseModel):
    status: str = Field("ready", description="Reward signal service operational status")
    service: str = Field("reward_signal", description="Service identifier")
    learning_active: bool = Field(False, description="True if online RL policy model update training is active")
