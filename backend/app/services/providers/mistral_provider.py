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

from app.services.cost_calculator import CostCalculator

logger = logging.getLogger("orchestrator")

class MistralProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model_id: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.MISTRAL_API_KEY
        self.default_model = model_id or getattr(settings, "MISTRAL_MODEL", "mistral-small-latest")

    def _is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and self.api_key != "your_mistral_api_key_here")

    def get_status(self) -> ProviderStatusResponse:
        configured = self._is_configured()
        return ProviderStatusResponse(
            provider="Mistral API",
            configured=configured,
            available=configured,
            status_message="MISTRAL_API_KEY is configured and active." if configured else "MISTRAL_API_KEY is not configured.",
            models=[self.default_model, "mistral-medium-latest", "mistral-large-latest"]
        )

    def generate(self, request: ProviderGenerationRequest) -> ProviderGenerationResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="Mistral API",
                model_id=request.model_id or self.default_model,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message="Prompt text cannot be empty."
            )

        if not self._is_configured():
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="Mistral API",
                model_id=request.model_id or self.default_model,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message="MISTRAL_API_KEY is not configured in environment variables."
            )

        target_model = request.model_id if request.model_id and "mistral" in request.model_id else self.default_model
        logger.info(f"Executing Mistral API call for model '{target_model}'...")

        # Option A: Try official mistralai SDK if installed
        try:
            from mistralai import Mistral
            client = Mistral(api_key=self.api_key)
            messages = [{"role": "user", "content": request.prompt}]
            if request.system_instruction:
                messages.insert(0, {"role": "system", "content": request.system_instruction})

            response = client.chat.complete(
                model=target_model,
                messages=messages,
                temperature=request.temperature or 0.7,
                max_tokens=request.max_output_tokens
            )

            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)

            generated_text = None
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = TokenUsage(
                    input_tokens=getattr(response.usage, "prompt_tokens", None),
                    output_tokens=getattr(response.usage, "completion_tokens", None),
                    total_tokens=getattr(response.usage, "total_tokens", None)
                )

            cost_info = CostCalculator.calculate_cost("Mistral API", target_model, usage, "online")
            logger.info(f"Mistral API call via SDK completed in {latency_ms} ms.")
            return ProviderGenerationResponse(
                provider="Mistral API",
                model_id=target_model,
                generated_text=generated_text,
                finish_reason="STOP" if generated_text else "UNKNOWN",
                usage=usage,
                cost=cost_info["cost"],
                cost_currency=cost_info["cost_currency"],
                cost_source=cost_info["cost_source"],
                latency_ms=latency_ms,
                success=True if generated_text else False,
                error_message=None if generated_text else "Mistral API returned empty content"
            )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Mistral SDK call failed, attempting direct HTTPS REST fallback: {str(e)}")

        # Option B: Direct HTTPS REST API call to https://api.mistral.ai/v1/chat/completions
        try:
            url = "https://api.mistral.ai/v1/chat/completions"
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

            cost_info = CostCalculator.calculate_cost("Mistral API", target_model, usage, "online")
            logger.info(f"Mistral REST API call completed in {latency_ms} ms.")
            return ProviderGenerationResponse(
                provider="Mistral API",
                model_id=target_model,
                generated_text=generated_text,
                finish_reason=choices[0].get("finish_reason", "STOP") if choices else "STOP",
                usage=usage,
                cost=cost_info["cost"],
                cost_currency=cost_info["cost_currency"],
                cost_source=cost_info["cost_source"],
                latency_ms=latency_ms,
                success=True if generated_text else False,
                error_message=None if generated_text else "Mistral API returned empty text"
            )

        except urllib.error.HTTPError as http_err:
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)
            err_body = ""
            try:
                err_body = http_err.read().decode("utf-8")
            except Exception:
                pass
            error_msg = f"Mistral API HTTP {http_err.code}: {http_err.reason}. {err_body[:100]}"
            logger.error(error_msg)
            return ProviderGenerationResponse(
                provider="Mistral API",
                model_id=target_model,
                generated_text=None,
                latency_ms=latency_ms,
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)
            error_msg = f"Mistral API execution error: {str(e)}"
            logger.error(error_msg)
            return ProviderGenerationResponse(
                provider="Mistral API",
                model_id=target_model,
                generated_text=None,
                latency_ms=latency_ms,
                success=False,
                error_message=error_msg
            )
