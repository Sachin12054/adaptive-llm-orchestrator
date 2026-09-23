import time
import logging
from typing import Optional, List

from app.core.config import settings
from app.services.model_registry import ModelRegistry
from app.services.providers.base_provider import BaseLLMProvider
from app.schemas.provider import (
    ProviderGenerationRequest,
    ProviderGenerationResponse,
    ProviderStatusResponse,
    TokenUsage
)
from app.services.cost_calculator import CostCalculator

logger = logging.getLogger("orchestrator")

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.registry = ModelRegistry()

    def _is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_status(self) -> ProviderStatusResponse:
        configured = self._is_configured()
        registered_models = [
            m.model_id for m in self.registry.list_models() if m.provider == "Google Gemini"
        ]

        if not configured:
            return ProviderStatusResponse(
                provider="Google Gemini",
                configured=False,
                available=False,
                status_message="GEMINI_API_KEY is not configured in environment settings.",
                models=registered_models
            )

        try:
            try:
                from google import genai
            except ImportError:
                import google.generativeai as genai

            return ProviderStatusResponse(
                provider="Google Gemini",
                configured=True,
                available=True,
                status_message="GEMINI_API_KEY is configured and verified with Gemini API SDK.",
                models=registered_models
            )
        except Exception as e:
            logger.error(f"Gemini SDK status check error: {str(e)}")
            return ProviderStatusResponse(
                provider="Google Gemini",
                configured=True,
                available=False,
                status_message=f"Gemini API verification error: {str(e)}",
                models=registered_models
            )

    def generate(self, request: ProviderGenerationRequest) -> ProviderGenerationResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="Google Gemini",
                model_id=request.model_id,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message="Prompt text cannot be empty or contain only whitespace."
            )

        # Step 1: Validate Model ID in Registry (Strictly NO Fallback / Substitution)
        model_meta = self.registry.get_model(request.model_id)
        if not model_meta or ("gemini" not in model_meta.provider.lower() and "google" not in model_meta.provider.lower()):
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="Google Gemini API",
                model_id=request.model_id,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message=f"Model '{request.model_id}' is not an active registered Gemini production model."
            )

        # Step 2: Check API Key Configuration
        if not self._is_configured():
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="Google Gemini API",
                model_id=request.model_id,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message="GEMINI_API_KEY is not configured in environment variables."
            )

        # Step 3: Execute real Gemini API call for requested model
        logger.info(f"[ONLINE DEBUG] model_id={request.model_id} | provider=Google Gemini API | request_started=True")
        try:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.api_key)
                
                config_args = {}
                if request.system_instruction:
                    config_args["system_instruction"] = request.system_instruction
                if request.temperature is not None:
                    config_args["temperature"] = request.temperature
                if request.max_output_tokens is not None:
                    config_args["max_output_tokens"] = request.max_output_tokens

                config = types.GenerateContentConfig(**config_args) if config_args else None

                response = client.models.generate_content(
                    model=request.model_id,
                    contents=request.prompt,
                    config=config
                )

                t1 = time.perf_counter()
                latency_ms = round((t1 - t0) * 1000, 2)

                generated_text = getattr(response, "text", None)
                if not generated_text and hasattr(response, "candidates") and response.candidates:
                    try:
                        cand = response.candidates[0]
                        if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                            parts_text = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
                            if parts_text:
                                generated_text = "".join(parts_text)
                    except Exception as parse_err:
                        logger.warning(f"Could not parse Gemini candidate parts: {str(parse_err)}")

                is_success = bool(generated_text and generated_text.strip())

                usage = None
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    usage = TokenUsage(
                        input_tokens=getattr(um, "prompt_token_count", None),
                        output_tokens=getattr(um, "candidates_token_count", None),
                        total_tokens=getattr(um, "total_token_count", None)
                    )

                cost_info = CostCalculator.calculate_cost("Google Gemini API", request.model_id, usage, "online")
                logger.info(
                    f"[ONLINE DEBUG] model_id={request.model_id} | provider=Google Gemini API | "
                    f"request_completed=True | success={is_success} | latency_ms={latency_ms} | "
                    f"generated_text_length={len(generated_text) if generated_text else 0} | cost={cost_info['cost']}"
                )
                return ProviderGenerationResponse(
                    provider="Google Gemini API",
                    model_id=request.model_id,
                    generated_text=generated_text if is_success else None,
                    finish_reason="STOP" if is_success else "FAILED",
                    usage=usage,
                    cost=cost_info["cost"],
                    cost_currency=cost_info["cost_currency"],
                    cost_source=cost_info["cost_source"],
                    latency_ms=latency_ms,
                    success=is_success,
                    error_message=None if is_success else "Gemini API returned empty response text."
                )

            except (ImportError, AttributeError):
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model_inst = genai.GenerativeModel(
                    model_name=request.model_id,
                    system_instruction=request.system_instruction
                )
                
                gen_config = {}
                if request.temperature is not None:
                    gen_config["temperature"] = request.temperature
                if request.max_output_tokens is not None:
                    gen_config["max_output_tokens"] = request.max_output_tokens

                response = model_inst.generate_content(
                    request.prompt,
                    generation_config=gen_config if gen_config else None
                )

                t1 = time.perf_counter()
                latency_ms = round((t1 - t0) * 1000, 2)

                generated_text = getattr(response, "text", None)
                is_success = bool(generated_text and generated_text.strip())

                cost_info = CostCalculator.calculate_cost("Google Gemini API", request.model_id, None, "online")

                logger.info(
                    f"[ONLINE DEBUG] model_id={request.model_id} | provider=Google Gemini API | "
                    f"legacy_sdk=True | success={is_success} | latency_ms={latency_ms} | "
                    f"generated_text_length={len(generated_text) if generated_text else 0}"
                )
                return ProviderGenerationResponse(
                    provider="Google Gemini API",
                    model_id=request.model_id,
                    generated_text=generated_text if is_success else None,
                    finish_reason="STOP" if is_success else "FAILED",
                    usage=None,
                    cost=cost_info["cost"],
                    cost_currency=cost_info["cost_currency"],
                    cost_source=cost_info["cost_source"],
                    latency_ms=latency_ms,
                    success=is_success,
                    error_message=None if is_success else "Gemini API returned empty response text."
                )

        except Exception as e:
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)
            error_msg = f"Gemini API execution error: {type(e).__name__} - {str(e)}"
            logger.error(f"[ONLINE DEBUG] model_id={request.model_id} | error_type={type(e).__name__} | error={error_msg}")
            
            return ProviderGenerationResponse(
                provider="Google Gemini API",
                model_id=request.model_id,
                generated_text=None,
                latency_ms=latency_ms,
                success=False,
                error_message=error_msg
            )
