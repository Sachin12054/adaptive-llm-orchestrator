from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from app.schemas.provider import TokenUsage

class ResponseGenerationRequest(BaseModel):
    prompt: str = Field(..., description="User prompt text to generate response for")
    selected_model: str = Field(..., description="Target model identifier selected upstream (e.g. qwen-coder-3b)")
    system_instruction: Optional[str] = Field(None, description="Optional system instruction / persona prompt")
    temperature: Optional[float] = Field(None, description="Optional sampling temperature [0.0 - 2.0]")
    max_output_tokens: Optional[int] = Field(None, description="Optional maximum output token limit")
    execution_mode: Optional[str] = Field("local", description="Execution mode: local (Ollama) or gemini (Google Gemini API)")

class ResponseGenerationResponse(BaseModel):
    success: bool = Field(..., description="True if text generation completed successfully")
    model_id: Optional[str] = Field(None, description="Target model identifier executed")
    provider: Optional[str] = Field(None, description="Provider adapter used for execution")
    generated_text: Optional[str] = Field(None, description="Generated response text returned by provider")
    finish_reason: Optional[str] = Field(None, description="Generation finish reason (e.g. STOP, LENGTH)")
    latency_ms: float = Field(..., description="Total end-to-end execution latency in milliseconds")
    ollama_load_ms: Optional[float] = Field(None, description="Ollama model load duration in milliseconds")
    ollama_prompt_eval_ms: Optional[float] = Field(None, description="Ollama prompt evaluation duration in milliseconds")
    ollama_eval_ms: Optional[float] = Field(None, description="Ollama token generation duration in milliseconds")
    usage: Optional[TokenUsage] = Field(None, description="Token usage metrics if provided by provider")
    execution_status: str = Field(..., description="Status: completed, failed, unsupported_model, not_configured")
    error_message: Optional[str] = Field(None, description="Error details if execution failed")

    # Cost Telemetry Fields
    cost: float = Field(0.0, description="Calculated monetary API execution cost in USD")
    cost_currency: str = Field("USD", description="Currency code (default USD)")
    cost_source: str = Field("zero_local", description="Source: zero_local, configured_pricing_estimate, provider_reported, unknown")
    
    # Failover Telemetry
    initial_model: Optional[str] = Field(None, description="Initial model selected prior to failover")
    failover_used: bool = Field(False, description="True if fallback model was executed following provider failure")
    attempts: int = Field(1, description="Total execution attempts")
    attempts_detail: Optional[list] = Field(default_factory=list, description="Per-attempt telemetry breakdown")

class ResponseStatusResponse(BaseModel):
    status: str = Field(..., description="Response generator status: ready, unconfigured")
    service: str = Field("response_generator", description="Service identifier")
    model_manager_available: bool = Field(..., description="True if ModelManager has executable models ready")
