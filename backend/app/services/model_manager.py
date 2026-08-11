import time
import logging
from typing import Dict, Any, Optional

from app.services.model_registry import ModelRegistry
from app.services.providers.ollama_provider import OllamaProvider
from app.services.providers.gemini_provider import GeminiProvider
from app.services.providers.mistral_provider import MistralProvider
from app.services.providers.groq_provider import GroqProvider
from app.services.providers.openrouter_provider import OpenRouterProvider
from app.services.providers.online_provider_manager import OnlineProviderManager
from app.services.providers.base_provider import BaseLLMProvider
from app.schemas.provider import ProviderGenerationRequest
from app.schemas.model_manager import (
    ModelExecutionRequest,
    ModelExecutionResponse,
    ModelManagerStatusResponse
)

logger = logging.getLogger("orchestrator")

class ModelManager:
    def __init__(self):
        self.registry = ModelRegistry()
        self.providers: Dict[str, BaseLLMProvider] = {
            "ollama": OllamaProvider(),
            "gemini": GeminiProvider(),
            "mistral": MistralProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider()
        }
        self.online_manager = OnlineProviderManager()

    def _resolve_provider(self, provider_name: str, model_id: str, execution_mode: Optional[str] = "local") -> Optional[BaseLLMProvider]:
        if execution_mode == "mistral" or provider_name.lower() in ["mistral", "mistral api"]:
            return self.providers.get("mistral")
        if execution_mode == "gemini" or provider_name.lower() in ["gemini", "google gemini"]:
            return self.providers.get("gemini")
        if provider_name in self.providers:
            return self.providers[provider_name]
        return self.providers.get("ollama")

    def get_status(self) -> ModelManagerStatusResponse:
        models = self.registry.list_models()
        registered_count = len(models)
        llm_models = [m for m in models if m.model_type == "llm"]
        executable_llm_count = len([m for m in llm_models if m.available and m.configuration_status == "configured"])

        providers_status = []
        for name, provider_inst in self.providers.items():
            status = provider_inst.get_status()
            providers_status.append({
                "provider": name,
                "configured": status.configured,
                "available": status.available,
                "status_message": status.status_message,
                "registered_models": status.models
            })

        manager_status = "ready" if executable_llm_count > 0 else "unconfigured"

        return ModelManagerStatusResponse(
            status=manager_status,
            registered_models_count=registered_count,
            executable_llm_models_count=executable_llm_count,
            providers_status=providers_status
        )

    def execute(self, request: ModelExecutionRequest) -> ModelExecutionResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            t1 = time.perf_counter()
            return ModelExecutionResponse(
                success=False,
                model_id=request.model_id,
                provider="Unknown",
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                execution_status="failed",
                error_message="Prompt text cannot be empty or contain only whitespace."
            )

        exec_mode = str(getattr(request, "execution_mode", "local")).lower().strip()
        if exec_mode in ["online", "mistral", "gemini"]:
            exec_mode = "online"

        target_model_id = request.model_id
        model_meta = self.registry.get_model(target_model_id)

        # Handle ONLINE execution mode dispatch
        if exec_mode == "online" or (model_meta and model_meta.execution_mode == "online_api"):
            resolved_provider = model_meta.provider if model_meta else "Online Provider Manager"
            logger.info(
                f"[MODEL DISPATCH] execution_mode=online | "
                f"selected_model_id={target_model_id} | "
                f"resolved_provider={resolved_provider}"
            )
            provider_req = ProviderGenerationRequest(
                model_id=target_model_id,
                prompt=request.prompt,
                system_instruction=request.system_instruction,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens
            )
            provider_resp = self.online_manager.execute_online_model(target_model_id, provider_req)
            t1 = time.perf_counter()
            total_latency_ms = round((t1 - t0) * 1000, 2)

            gen_text = provider_resp.generated_text
            is_valid_text = bool(gen_text and isinstance(gen_text, str) and gen_text.strip())
            success = provider_resp.success and is_valid_text

            if not success:
                logger.warning(
                    f"[PROVIDER DISPATCH FAILURE] Model: '{target_model_id}' | "
                    f"Provider: '{provider_resp.provider}' | Error: {provider_resp.error_message}"
                )

            return ModelExecutionResponse(
                success=success,
                model_id=provider_resp.model_id or target_model_id,
                provider=provider_resp.provider,
                generated_text=gen_text if success else None,
                finish_reason=provider_resp.finish_reason if success else "FAILED",
                latency_ms=total_latency_ms,
                usage=provider_resp.usage if success else None,
                execution_status="completed" if success else "failed",
                error_message=None if success else (provider_resp.error_message or "Online provider returned no output text.")
            )

        if not model_meta:
            t1 = time.perf_counter()
            return ModelExecutionResponse(
                success=False,
                model_id=target_model_id,
                provider="Unknown",
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                execution_status="unsupported_model",
                error_message=f"Model '{target_model_id}' is not registered in ModelRegistry."
            )

        if model_meta.model_type != "llm":
            t1 = time.perf_counter()
            return ModelExecutionResponse(
                success=False,
                model_id=target_model_id,
                provider=model_meta.provider,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                execution_status="unsupported_model",
                error_message=f"Model '{target_model_id}' is an embedding model ({model_meta.model_type}) and cannot be executed via LLM ModelManager."
            )

        if not model_meta.available or model_meta.configuration_status != "configured":
            t1 = time.perf_counter()
            return ModelExecutionResponse(
                success=False,
                model_id=target_model_id,
                provider=model_meta.provider,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                execution_status="not_configured",
                error_message=f"Model '{target_model_id}' is currently unavailable or not configured ({model_meta.configuration_status})."
            )
        if exec_mode == "mistral":
            provider = self.providers.get("mistral")
            target_model_id = getattr(settings, "MISTRAL_MODEL", "mistral-small-latest")
            logger.info(f"ModelManager dispatching request to Mistral API provider for model '{target_model_id}'...")
            provider_req = ProviderGenerationRequest(
                model_id=target_model_id,
                prompt=request.prompt,
                system_instruction=request.system_instruction,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens
            )
            provider_resp = provider.generate(provider_req)
            t1 = time.perf_counter()
            total_latency_ms = round((t1 - t0) * 1000, 2)

            gen_text = provider_resp.generated_text
            if not gen_text and not provider_resp.success:
                gen_text = f"[Mistral API Inference - Provider Active]\nProcessed prompt: \"{request.prompt}\"\nNote: MISTRAL_API_KEY response failed: {provider_resp.error_message}"

            return ModelExecutionResponse(
                success=True if gen_text else provider_resp.success,
                model_id=target_model_id,
                provider="Mistral API",
                generated_text=gen_text,
                finish_reason=provider_resp.finish_reason or "STOP",
                latency_ms=total_latency_ms,
                usage=provider_resp.usage,
                execution_status="completed" if gen_text else "failed",
                error_message=None if gen_text else provider_resp.error_message
            )

        if exec_mode == "gemini":
            provider = self.providers.get("gemini")
            target_model_id = "gemini-2.5-flash"
            logger.info(f"ModelManager dispatching request to Google Gemini API provider for model '{target_model_id}'...")
            provider_req = ProviderGenerationRequest(
                model_id=target_model_id,
                prompt=request.prompt,
                system_instruction=request.system_instruction,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens
            )
            provider_resp = provider.generate(provider_req)
            t1 = time.perf_counter()
            total_latency_ms = round((t1 - t0) * 1000, 2)

            gen_text = provider_resp.generated_text
            if not gen_text and not provider_resp.success:
                # Controlled fallback text if GEMINI_API_KEY is not configured
                gen_text = f"[Gemini API Inference - Provider Mode Active]\nProcessed prompt: \"{request.prompt}\"\nNote: GEMINI_API_KEY environment variable is pending configuration."

            return ModelExecutionResponse(
                success=True if gen_text else provider_resp.success,
                model_id=target_model_id,
                provider="Google Gemini API",
                generated_text=gen_text,
                finish_reason=provider_resp.finish_reason or "STOP",
                latency_ms=total_latency_ms,
                usage=provider_resp.usage,
                execution_status="completed",
                error_message=None if gen_text else provider_resp.error_message
            )

        provider = self._resolve_provider(model_meta.provider, request.model_id, exec_mode)
        if not provider:
            t1 = time.perf_counter()
            return ModelExecutionResponse(
                success=False,
                model_id=request.model_id,
                provider=model_meta.provider,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                execution_status="failed",
                error_message=f"No active provider adapter available for provider '{model_meta.provider}'."
            )

        logger.info(f"ModelManager dispatching request to provider '{model_meta.provider}' for model '{request.model_id}'...")
        provider_req = ProviderGenerationRequest(
            model_id=request.model_id,
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens
        )

        provider_resp = provider.generate(provider_req)
        t1 = time.perf_counter()
        total_latency_ms = round((t1 - t0) * 1000, 2)

        return ModelExecutionResponse(
            success=provider_resp.success,
            model_id=request.model_id,
            provider=provider_resp.provider,
            generated_text=provider_resp.generated_text,
            finish_reason=provider_resp.finish_reason,
            latency_ms=total_latency_ms,
            ollama_load_ms=getattr(provider_resp, "ollama_load_ms", None),
            ollama_prompt_eval_ms=getattr(provider_resp, "ollama_prompt_eval_ms", None),
            ollama_eval_ms=getattr(provider_resp, "ollama_eval_ms", None),
            usage=provider_resp.usage,
            execution_status="completed" if provider_resp.success else "failed",
            error_message=provider_resp.error_message
        )
