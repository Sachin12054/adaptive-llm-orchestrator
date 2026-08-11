import time
import json
import logging
import urllib.request
import urllib.error
from typing import Optional

from app.core.config import settings
from app.services.providers.base_provider import BaseLLMProvider
from app.schemas.provider import (
    ProviderGenerationRequest,
    ProviderGenerationResponse,
    ProviderStatusResponse,
    TokenUsage
)

logger = logging.getLogger("orchestrator")

class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.GROQ_API_KEY
        self.default_model = model_id or getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")

    def _is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and self.api_key != "your_groq_api_key_here")

    def get_status(self) -> ProviderStatusResponse:
        configured = self._is_configured()
        return ProviderStatusResponse(
            provider="Groq API",
            configured=configured,
            available=configured,
            status_message="GROQ_API_KEY is configured and active." if configured else "GROQ_API_KEY is not configured.",
            models=[self.default_model, "mixtral-8x7b-32768", "gemma2-9b-it"]
        )

    def generate(self, request: ProviderGenerationRequest) -> ProviderGenerationResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="Groq API",
                model_id=request.model_id or self.default_model,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message="Prompt text cannot be empty."
            )

        if not self._is_configured():
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="Groq API",
                model_id=request.model_id or self.default_model,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message="GROQ_API_KEY is not configured in environment variables."
            )

        target_model = request.model_id if request.model_id and ("llama" in request.model_id or "mixtral" in request.model_id or "groq" in request.model_id) else self.default_model
        logger.info(f"Executing Groq API call for model '{target_model}'...")

        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            messages = [{"role": "user", "content": request.prompt}]
            if request.system_instruction:
                messages.insert(0, {"role": "system", "content": request.system_instruction})

            payload = {
                "model": target_model,
                "messages": messages,
                "temperature": request.temperature or 0.7
            }
            if request.max_output_tokens:
                payload["max_tokens"] = request.max_output_tokens

            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_bytes = resp.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))

            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)

            generated_text = None
            choices = resp_json.get("choices", [])
            if choices and len(choices) > 0:
                msg = choices[0].get("message", {})
                generated_text = msg.get("content")

            usage_raw = resp_json.get("usage", {})
            usage = TokenUsage(
                input_tokens=usage_raw.get("prompt_tokens"),
                output_tokens=usage_raw.get("completion_tokens"),
                total_tokens=usage_raw.get("total_tokens")
            )

            logger.info(f"Groq API call completed in {latency_ms} ms.")
            return ProviderGenerationResponse(
                provider="Groq API",
                model_id=target_model,
                generated_text=generated_text,
                finish_reason=choices[0].get("finish_reason", "STOP") if choices else "STOP",
                usage=usage,
                latency_ms=latency_ms,
                success=True if generated_text else False,
                error_message=None if generated_text else "Groq API returned empty text"
            )

        except urllib.error.HTTPError as http_err:
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)
            err_body = ""
            try:
                err_body = http_err.read().decode("utf-8")
            except Exception:
                pass
            error_msg = f"Groq API HTTP {http_err.code}: {http_err.reason}. {err_body[:100]}"
            logger.error(error_msg)
            return ProviderGenerationResponse(
                provider="Groq API",
                model_id=target_model,
                generated_text=None,
                latency_ms=latency_ms,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)
            error_msg = f"Groq API execution error: {str(e)}"
            logger.error(error_msg)
            return ProviderGenerationResponse(
                provider="Groq API",
                model_id=target_model,
                generated_text=None,
                latency_ms=latency_ms,
                success=False,
                error_message=error_msg
            )
