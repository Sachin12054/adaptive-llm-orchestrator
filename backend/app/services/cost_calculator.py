import logging
from typing import Dict, Any, Optional
from app.schemas.provider import TokenUsage

logger = logging.getLogger("orchestrator")

# Centralized Model Pricing Table (per 1,000 tokens in USD)
# GROUNDED STRICTLY IN REGISTERED PROJECT MODELS:
# - Ollama Local Models: Gemma 3 4B, Qwen Coder 3B, DeepSeek R1 7B -> $0.00
# - Google Gemini: gemini-3.5-flash, gemini-2.5-flash -> Input $0.00015 / Output $0.00060
# - Mistral API: mistral-small-latest, mistral-medium-latest, mistral-large-latest -> Input $0.00020 / Output $0.00060
# - Groq API: llama-3.3-70b-versatile, mixtral-8x7b-32768 -> Input $0.00059 / Output $0.00079
# - OpenRouter API: meta-llama/llama-3.3-70b-instruct -> Input $0.00040 / Output $0.00040
PRICING_TABLE: Dict[str, Dict[str, float]] = {
    # Gemini Models
    "gemini-3.5-flash": {"input_cost_per_1k": 0.00015, "output_cost_per_1k": 0.00060},
    "gemini-2.5-flash": {"input_cost_per_1k": 0.00015, "output_cost_per_1k": 0.00060},
    "gemini-2.5-pro": {"input_cost_per_1k": 0.00125, "output_cost_per_1k": 0.00500},

    # Mistral Models
    "mistral-small-latest": {"input_cost_per_1k": 0.00020, "output_cost_per_1k": 0.00060},
    "mistral-medium-latest": {"input_cost_per_1k": 0.00070, "output_cost_per_1k": 0.00210},
    "mistral-large-latest": {"input_cost_per_1k": 0.00200, "output_cost_per_1k": 0.00600},

    # Groq Models
    "llama-3.3-70b-versatile": {"input_cost_per_1k": 0.00059, "output_cost_per_1k": 0.00079},
    "mixtral-8x7b-32768": {"input_cost_per_1k": 0.00027, "output_cost_per_1k": 0.00027},
    "gemma2-9b-it": {"input_cost_per_1k": 0.00020, "output_cost_per_1k": 0.00020},

    # OpenRouter Models
    "meta-llama/llama-3.3-70b-instruct": {"input_cost_per_1k": 0.00040, "output_cost_per_1k": 0.00040},

    # Local Ollama Models (Explicitly 0.0)
    "gemma-3-4b": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
    "qwen-coder-3b": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
    "deepseek-r1-7b": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
    "BAAI/bge-m3": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
}

class CostCalculator:
    """
    Centralized Cost Accounting Engine.
    Computes exact or estimated monetary costs for LLM inference requests.
    Supports local execution ($0.00), online cloud API estimates, and zero-overhead synthesis.
    """

    @staticmethod
    def calculate_cost(
        provider: str,
        model_id: str,
        usage: Optional[TokenUsage] = None,
        execution_mode: Optional[str] = "local",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculates request cost based on provider, model, token usage, and execution mode.
        """
        exec_mode = str(execution_mode or "local").lower().strip()
        provider_lower = str(provider or "").lower().strip()

        # Extract tokens from usage object or direct parameters
        in_tok = None
        out_tok = None
        tot_tok = None

        if usage:
            in_tok = usage.input_tokens
            out_tok = usage.output_tokens
            tot_tok = usage.total_tokens

        if in_tok is None and input_tokens is not None:
            in_tok = input_tokens
        if out_tok is None and output_tokens is not None:
            out_tok = output_tokens
        if tot_tok is None and in_tok is not None and out_tok is not None:
            tot_tok = in_tok + out_tok

        # Case 1: Local Execution (Ollama)
        if exec_mode == "local" or provider_lower == "ollama" or "ollama" in provider_lower:
            return {
                "cost": 0.0,
                "cost_currency": "USD",
                "cost_source": "zero_local",
                "input_cost": 0.0,
                "output_cost": 0.0,
                "input_tokens": in_tok or 0,
                "output_tokens": out_tok or 0,
                "total_tokens": tot_tok or 0
            }

        # Case 2: Cloud / Online Provider Execution
        pricing = PRICING_TABLE.get(model_id)
        if not pricing:
            # Fallback lookup by prefix / provider
            if "gemini" in model_id.lower():
                pricing = PRICING_TABLE["gemini-3.5-flash"]
            elif "mistral" in model_id.lower():
                pricing = PRICING_TABLE["mistral-small-latest"]
            elif "groq" in model_id.lower() or "llama-3.3-70b" in model_id.lower():
                pricing = PRICING_TABLE["llama-3.3-70b-versatile"]
            else:
                pricing = {"input_cost_per_1k": 0.00030, "output_cost_per_1k": 0.00060}

        if in_tok is None and out_tok is None:
            return {
                "cost": 0.0,
                "cost_currency": "USD",
                "cost_source": "unknown",
                "input_cost": 0.0,
                "output_cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }

        in_tok_val = in_tok or 0
        out_tok_val = out_tok or 0
        tot_tok_val = tot_tok or (in_tok_val + out_tok_val)

        input_cost = (in_tok_val / 1000.0) * pricing["input_cost_per_1k"]
        output_cost = (out_tok_val / 1000.0) * pricing["output_cost_per_1k"]
        total_cost = round(input_cost + output_cost, 6)

        return {
            "cost": total_cost,
            "cost_currency": "USD",
            "cost_source": "configured_pricing_estimate",
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "input_tokens": in_tok_val,
            "output_tokens": out_tok_val,
            "total_tokens": tot_tok_val
        }
