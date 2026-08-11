from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.provider import TokenUsage

class ModelExecutionRequest(BaseModel):
    model_id: str = Field(..., description="Target model identifier to execute (e.g. qwen-coder-3b, gemma-3-4b)")
    prompt: str = Field(..., description="User prompt text to generate response for")
    system_instruction: Optional[str] = Field(None, description="Optional system instruction / persona prompt")
    temperature: Optional[float] = Field(None, description="Optional sampling temperature [0.0 - 2.0]")
    max_output_tokens: Optional[int] = Field(None, description="Optional maximum output token limit")
    execution_mode: Optional[str] = Field("local", description="Execution mode: local or gemini")

class ModelExecutionResponse(BaseModel):
    success: bool = Field(..., description="True if text generation completed successfully")
    model_id: str = Field(..., description="Target model identifier executed")
    provider: str = Field(..., description="Provider adapter used for execution")
    generated_text: Optional[str] = Field(None, description="Generated response text returned by provider")
    finish_reason: Optional[str] = Field(None, description="Generation finish reason (e.g. STOP, LENGTH)")
    latency_ms: float = Field(..., description="Total end-to-end execution latency in milliseconds")
    ollama_load_ms: Optional[float] = Field(None, description="Ollama model load duration in milliseconds")
    ollama_prompt_eval_ms: Optional[float] = Field(None, description="Ollama prompt evaluation duration in milliseconds")
    ollama_eval_ms: Optional[float] = Field(None, description="Ollama token generation duration in milliseconds")
    usage: Optional[TokenUsage] = Field(None, description="Token usage metrics if provided by provider")
    execution_status: str = Field(..., description="Status: completed, failed, unsupported_model, not_configured")
    error_message: Optional[str] = Field(None, description="Error details if execution failed")

class ModelManagerStatusResponse(BaseModel):
    status: str = Field(..., description="Manager operational status: ready, partially_configured, unconfigured")
    registered_models_count: int = Field(..., description="Total count of models in ModelRegistry")
    executable_llm_models_count: int = Field(..., description="Count of executable LLM models")
    providers_status: List[Dict[str, Any]] = Field(..., description="Status breakdown of configured provider adapters")
