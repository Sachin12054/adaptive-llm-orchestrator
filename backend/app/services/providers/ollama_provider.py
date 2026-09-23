import time
import json
import logging
import urllib.request
import urllib.error
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

class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.registry = ModelRegistry()

    def get_status(self) -> ProviderStatusResponse:
        registered_models = [
            m.model_id for m in self.registry.list_models()
            if m.provider.lower() == "ollama" or m.execution_mode == "local"
        ]

        tags_url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(tags_url, headers={"User-Agent": "AdaptiveOrchestrator/0.1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    raw_data = resp.read().decode("utf-8")
                    data = json.loads(raw_data)
                    installed_models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                    
                    return ProviderStatusResponse(
                        provider="ollama",
                        configured=True,
                        available=True,
                        status_message=f"Ollama local server is running on {self.base_url} ({len(installed_models)} models installed).",
                        models=registered_models
                    )
        except Exception as e:
            logger.info(f"Ollama server check on {self.base_url} failed: {str(e)}")

        return ProviderStatusResponse(
            provider="ollama",
            configured=False,
            available=False,
            status_message=f"Ollama local server is not running or unreachable on {self.base_url}.",
            models=registered_models
        )

    def generate(self, request: ProviderGenerationRequest) -> ProviderGenerationResponse:
        t0 = time.perf_counter()

        if not request.prompt or not request.prompt.strip():
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="ollama",
                model_id=request.model_id,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message="Prompt text cannot be empty or contain only whitespace."
            )

        # Step 1: Look up model metadata in ModelRegistry
        model_meta = self.registry.get_model(request.model_id)
        if not model_meta or (model_meta.provider.lower() != "ollama" and model_meta.execution_mode != "local"):
            t1 = time.perf_counter()
            return ProviderGenerationResponse(
                provider="ollama",
                model_id=request.model_id,
                generated_text=None,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                error_message=f"Model '{request.model_id}' is not an active registered Ollama model."
            )

        # Step 2: Construct payload for local Ollama /api/generate endpoint
        generate_url = f"{self.base_url}/api/generate"
        payload = {
            "model": request.model_id,
            "prompt": request.prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": request.temperature if request.temperature is not None else 0.2,
                "num_predict": request.max_output_tokens if request.max_output_tokens is not None else 1024,
                "num_ctx": 4096
            }
        }

        # Include system instruction ONLY if explicitly provided
        if request.system_instruction and request.system_instruction.strip():
            payload["system"] = request.system_instruction.strip()

        logger.info(f"Executing local Ollama API call for model '{request.model_id}' on {generate_url}...")

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                generate_url,
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "AdaptiveOrchestrator/0.1.0"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=300.0) as resp:
                t1 = time.perf_counter()
                latency_ms = round((t1 - t0) * 1000, 2)

                if resp.status != 200:
                    return ProviderGenerationResponse(
                        provider="ollama",
                        model_id=request.model_id,
                        generated_text=None,
                        latency_ms=latency_ms,
                        success=False,
                        error_message=f"Ollama server returned HTTP error status {resp.status}."
                    )

                raw_resp = resp.read().decode("utf-8")
                res_data = json.loads(raw_resp)

                generated_text = res_data.get("response")
                if not generated_text:
                    return ProviderGenerationResponse(
                        provider="ollama",
                        model_id=request.model_id,
                        generated_text=None,
                        latency_ms=latency_ms,
                        success=False,
                        error_message="Ollama returned an empty response text."
                    )

                # Filter out DeepSeek <think>...</think> internal reasoning blocks from public response text
                import re
                generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()

                prompt_eval_count = res_data.get("prompt_eval_count", 0)
                eval_count = res_data.get("eval_count", 0)
                usage = None
                if prompt_eval_count or eval_count:
                    usage = TokenUsage(
                        input_tokens=prompt_eval_count,
                        output_tokens=eval_count,
                        total_tokens=prompt_eval_count + eval_count
                    )

                # High-resolution internal Ollama timings (convert ns to ms)
                load_ms = round(res_data.get("load_duration", 0) / 1e6, 2) if "load_duration" in res_data else None
                prompt_eval_ms = round(res_data.get("prompt_eval_duration", 0) / 1e6, 2) if "prompt_eval_duration" in res_data else None
                eval_ms = round(res_data.get("eval_duration", 0) / 1e6, 2) if "eval_duration" in res_data else None
                eval_duration_sec = (res_data.get("eval_duration", 0) / 1e9) if "eval_duration" in res_data else 0.0

                tokens_per_sec = round(eval_count / eval_duration_sec, 2) if eval_duration_sec > 0 and eval_count > 0 else 0.0

                # System GPU Diagnostics
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_device_name = torch.cuda.get_device_name(0)
                        gpu_status = f"Model Running on GPU ({gpu_device_name})" if (tokens_per_sec > 10.0 or (load_ms is not None and load_ms < 3000.0)) else f"GPU Available ({gpu_device_name})"
                    else:
                        gpu_status = "Model Running on CPU (No CUDA)"
                except Exception:
                    gpu_status = "Model Running on Local Host"

                logger.info(
                    f"[OLLAMA DIAGNOSTICS] Model: '{request.model_id}' | "
                    f"Load: {load_ms}ms | Eval: {eval_ms}ms | Total: {latency_ms}ms | "
                    f"Prompt Tokens: {prompt_eval_count} | Generated Tokens: {eval_count} | "
                    f"Speed: {tokens_per_sec} tokens/sec | Execution: {gpu_status}"
                )

                cost_info = CostCalculator.calculate_cost("ollama", request.model_id, usage, "local")
                return ProviderGenerationResponse(
                    provider="ollama",
                    model_id=request.model_id,
                    generated_text=generated_text,
                    finish_reason="STOP",
                    usage=usage,
                    cost=cost_info["cost"],
                    cost_currency=cost_info["cost_currency"],
                    cost_source=cost_info["cost_source"],
                    latency_ms=latency_ms,
                    ollama_load_ms=load_ms,
                    ollama_prompt_eval_ms=prompt_eval_ms,
                    ollama_eval_ms=eval_ms,
                    success=True,
                    error_message=None
                )

        except urllib.error.HTTPError as e:
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            
            if e.code == 404 or "not found" in error_body.lower():
                err_msg = f"Model '{request.model_id}' is not installed in local Ollama instance."
            else:
                err_msg = f"Ollama HTTP Error {e.code}: {error_body or str(e)}"
            
            logger.error(f"Ollama HTTP error for model '{request.model_id}': {err_msg}")
            return ProviderGenerationResponse(
                provider="ollama",
                model_id=request.model_id,
                generated_text=None,
                latency_ms=latency_ms,
                success=False,
                error_message=err_msg
            )

        except Exception as e:
            t1 = time.perf_counter()
            latency_ms = round((t1 - t0) * 1000, 2)
            err_msg = f"Ollama local server connection error on {self.base_url}: {str(e)}"
            logger.error(err_msg)
            return ProviderGenerationResponse(
                provider="ollama",
                model_id=request.model_id,
                generated_text=None,
                latency_ms=latency_ms,
                success=False,
                error_message=err_msg
            )
