from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ProviderGenerationRequest(BaseModel):
    model_id: str = Field(..., description="Target model identifier (e.g. qwen-coder-3b, gemma-3-4b)")
    prompt: str = Field(..., description="User prompt text to generate response for")
    system_instruction: Optional[str] = Field(None, description="Optional system instruction / persona prompt")
    temperature: Optional[float] = Field(None, description="Optional sampling temperature [0.0 - 2.0]")
    max_output_tokens: Optional[int] = Field(None, description="Optional maximum output tokens token limit")

class TokenUsage(BaseModel):
    input_tokens: Optional[int] = Field(None, description="Number of prompt tokens processed")
    output_tokens: Optional[int] = Field(None, description="Number of candidate tokens generated")
    total_tokens: Optional[int] = Field(None, description="Total token count")

class ProviderGenerationResponse(BaseModel):
    provider: str = Field("ollama", description="Provider vendor name")
    model_id: str = Field(..., description="Target model identifier used for generation")
    generated_text: Optional[str] = Field(None, description="Normalized response text returned by provider")
    finish_reason: Optional[str] = Field(None, description="Provider generation finish reason if reported")
    usage: Optional[TokenUsage] = Field(None, description="Token usage metrics if provided by API")
    latency_ms: float = Field(..., description="End-to-end API generation latency in milliseconds")
    ollama_load_ms: Optional[float] = Field(None, description="Ollama model load duration in milliseconds")
    ollama_prompt_eval_ms: Optional[float] = Field(None, description="Ollama prompt evaluation duration in milliseconds")
    ollama_eval_ms: Optional[float] = Field(None, description="Ollama token generation duration in milliseconds")
    success: bool = Field(..., description="True if text generation completed successfully")
    error_message: Optional[str] = Field(None, description="Error details if generation request failed")
    
    # Cost Accounting Fields
    cost: float = Field(0.0, description="Calculated monetary API execution cost in USD")
    cost_currency: str = Field("USD", description="Currency code (default USD)")
    cost_source: str = Field("zero_local", description="Source of cost calculation: zero_local, configured_pricing_estimate, provider_reported, unknown")

class ProviderStatusResponse(BaseModel):
    provider: str = Field("ollama", description="Provider vendor name")
    configured: bool = Field(..., description="True if provider is present in environment")
    available: bool = Field(..., description="True if provider client can connect and generate content")
    status_message: str = Field(..., description="Human-readable status summary")
    models: List[str] = Field(..., description="List of models registered in ModelRegistry")
