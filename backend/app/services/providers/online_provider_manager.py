import logging
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.schemas.model import ModelMetadata
from app.schemas.provider import ProviderGenerationRequest, ProviderGenerationResponse
from app.services.providers.base_provider import BaseLLMProvider
from app.services.providers.gemini_provider import GeminiProvider
from app.services.providers.mistral_provider import MistralProvider
from app.services.providers.groq_provider import GroqProvider
from app.services.providers.openrouter_provider import OpenRouterProvider

logger = logging.getLogger("orchestrator")

class OnlineProviderManager:
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {
            "gemini": GeminiProvider(),
            "mistral": MistralProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider()
        }

    def get_online_model_candidates(self) -> List[ModelMetadata]:
        """
        Discovers available online cloud model candidates based ONLY on API key configuration.
        Does NOT perform task routing decisions - candidates are returned for BaselineAdaptivePolicy evaluation.
        """
        candidates: List[ModelMetadata] = []

        # 1. Gemini Model Candidate
        gemini_status = self.providers["gemini"].get_status()
        gemini_model_id = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
        candidates.append(ModelMetadata(
            model_id=gemini_model_id,
            provider="Google Gemini API",
            display_name="Gemini 2.5 Flash",
            model_type="llm",
            capabilities=["general_qa", "reasoning", "explanation", "coding", "summarization"],
            context_length=1000000,
            execution_mode="online_api",
            local=False,
            available=gemini_status.configured,
            configuration_status="configured" if gemini_status.configured else "not_configured",
            requirements={"env": "GEMINI_API_KEY"},
            metadata_source="online_provider_manager"
        ))

        # 2. Mistral Model Candidate
        mistral_status = self.providers["mistral"].get_status()
        mistral_model_id = getattr(settings, "MISTRAL_MODEL", "mistral-small-latest")
        candidates.append(ModelMetadata(
            model_id=mistral_model_id,
            provider="Mistral API",
            display_name="Mistral Small Latest",
            model_type="llm",
            capabilities=["general_qa", "reasoning", "explanation", "coding", "translation"],
            context_length=32000,
            execution_mode="online_api",
            local=False,
            available=mistral_status.configured,
            configuration_status="configured" if mistral_status.configured else "not_configured",
            requirements={"env": "MISTRAL_API_KEY"},
            metadata_source="online_provider_manager"
        ))

        # 3. Groq Model Candidate
        groq_status = self.providers["groq"].get_status()
        groq_model_id = getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
        candidates.append(ModelMetadata(
            model_id=groq_model_id,
            provider="Groq API",
            display_name="Groq LLaMA 3.3 70B",
            model_type="llm",
            capabilities=["coding", "general_qa", "reasoning", "mathematics"],
            context_length=128000,
            execution_mode="online_api",
            local=False,
            available=groq_status.configured,
            configuration_status="configured" if groq_status.configured else "not_configured",
            requirements={"env": "GROQ_API_KEY"},
            metadata_source="online_provider_manager"
        ))

        # 4. OpenRouter Model Candidate
        openrouter_status = self.providers["openrouter"].get_status()
        openrouter_model_id = getattr(settings, "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        candidates.append(ModelMetadata(
            model_id=openrouter_model_id,
            provider="OpenRouter API",
            display_name="OpenRouter LLaMA 3.3 70B",
            model_type="llm",
            capabilities=["reasoning", "general_qa", "coding", "explanation"],
            context_length=128000,
            execution_mode="online_api",
            local=False,
            available=openrouter_status.configured,
            configuration_status="configured" if openrouter_status.configured else "not_configured",
            requirements={"env": "OPENROUTER_API_KEY"},
            metadata_source="online_provider_manager"
        ))

        return candidates

    def execute_online_model(self, model_id: str, request: ProviderGenerationRequest) -> ProviderGenerationResponse:
        """Dispatches text generation for a specified online model to its concrete provider adapter."""
        if "gemini" in model_id.lower() or request.model_id == "gemini-2.5-flash":
            return self.providers["gemini"].generate(request)
        elif "mistral" in model_id.lower():
            return self.providers["mistral"].generate(request)
        elif "groq" in model_id.lower() or "llama-3.3-70b-versatile" in model_id.lower():
            return self.providers["groq"].generate(request)
        elif "openrouter" in model_id.lower() or "/" in model_id:
            return self.providers["openrouter"].generate(request)

        # Default fallback dispatch
        for name, provider in self.providers.items():
            if provider.get_status().configured:
                return provider.generate(request)

        return ProviderGenerationResponse(
            provider="Online Provider Manager",
            model_id=model_id,
            generated_text=None,
            latency_ms=0.0,
            success=False,
            error_message="No configured online provider available."
        )
