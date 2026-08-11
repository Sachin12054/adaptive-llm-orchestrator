from typing import Optional, List
from pydantic import BaseModel, Field

class VerificationRequest(BaseModel):
    prompt: str = Field(..., description="Original user prompt text")
    selected_model: str = Field(..., description="Target model identifier evaluated")
    generated_text: Optional[str] = Field(None, description="Generated response text to verify")
    generation_success: bool = Field(..., description="True if generation completed successfully")
    execution_status: str = Field(..., description="Execution status from ModelManager/ResponseGenerator")

class VerificationResponse(BaseModel):
    verified: bool = Field(..., description="True if response passes baseline structural and relevance checks")
    verification_status: str = Field(..., description="Status: verified_baseline, not_verifiable, empty_response, failed_checks")
    prompt: str = Field(..., description="Original user prompt text evaluated")
    selected_model: str = Field(..., description="Target model identifier evaluated")
    response_present: bool = Field(..., description="True if non-empty response text is present")
    relevance_score: float = Field(..., description="Prompt-response keyword/topic relevance score [0.0 - 1.0]")
    completeness_score: float = Field(..., description="Response completeness score [0.0 - 1.0]")
    structural_quality_score: float = Field(..., description="Structural formatting and validity score [0.0 - 1.0]")
    factual_verification_status: str = Field("not_verified", description="Factual verification status (always 'not_verified' in baseline)")
    issues: List[str] = Field(default_factory=list, description="List of detected verification issues or errors")
    verification_reasoning: List[str] = Field(..., description="Bulleted verification justification items")
    verification_latency_ms: float = Field(..., description="Verification evaluation latency in milliseconds")

class VerificationStatusResponse(BaseModel):
    status: str = Field("ready", description="Verifier operational status")
    service: str = Field("response_verifier", description="Service identifier")
    factual_verification_available: bool = Field(False, description="True if external factual retrieval verification is active")
