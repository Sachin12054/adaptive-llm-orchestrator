import time
import logging
from typing import Optional

from app.services.model_manager import ModelManager
from app.schemas.model_manager import ModelExecutionRequest
from app.schemas.response import (
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    ResponseStatusResponse
)

logger = logging.getLogger("orchestrator")

class ResponseGenerator:
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()

    def get_status(self) -> ResponseStatusResponse:
        mm_status = self.model_manager.get_status()
        is_ready = (mm_status.executable_llm_models_count > 0)
        
        return ResponseStatusResponse(
            status="ready" if is_ready else "unconfigured",
            service="response_generator",
            model_manager_available=is_ready
        )

    def generate_response(self, request: ResponseGenerationRequest) -> ResponseGenerationResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            raise ValueError("Prompt text for response generation cannot be empty or contain only whitespace.")

        logger.info(f"ResponseGenerator orchestrating generation for model '{request.selected_model}'...")

        exec_request = ModelExecutionRequest(
            model_id=request.selected_model,
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            execution_mode=getattr(request, "execution_mode", "local")
        )

        exec_response = self.model_manager.execute(exec_request)
        t1 = time.perf_counter()
        total_latency_ms = round((t1 - t0) * 1000, 2)

        return ResponseGenerationResponse(
            success=exec_response.success,
            model_id=exec_response.model_id,
            provider=exec_response.provider,
            generated_text=exec_response.generated_text,
            finish_reason=exec_response.finish_reason,
            latency_ms=max(exec_response.latency_ms, total_latency_ms),
            ollama_load_ms=getattr(exec_response, "ollama_load_ms", None),
            ollama_prompt_eval_ms=getattr(exec_response, "ollama_prompt_eval_ms", None),
            ollama_eval_ms=getattr(exec_response, "ollama_eval_ms", None),
            usage=exec_response.usage,
            cost=getattr(exec_response, "cost", 0.0),
            cost_currency=getattr(exec_response, "cost_currency", "USD"),
            cost_source=getattr(exec_response, "cost_source", "zero_local"),
            execution_status=exec_response.execution_status,
            error_message=exec_response.error_message
        )
