from typing import Dict, Any

# Token Pricing Table per 1,000 tokens
PRICING_TABLE = {
    "Local Ollama": {"input": 0.0, "output": 0.0},
    "Google Gemini API": {"input": 0.000075, "output": 0.000300},
    "Mistral API": {"input": 0.000100, "output": 0.000300},
    "Groq API": {"input": 0.000590, "output": 0.000790},
    "OpenRouter API": {"input": 0.000400, "output": 0.000800}
}

class CostMetricsEvaluator:
    """
    Cost Evaluation Engine.
    Calculates dollar cost per request based on model/provider token rates.
    Local Ollama models are explicitly reported as API Cost = $0.00 (with local compute telemetry).
    """

    @staticmethod
    def calculate_cost(
        provider: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, Any]:
        pricing = PRICING_TABLE.get(provider, {"input": 0.000200, "output": 0.000500})
        
        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "is_local_zero_dollar_cost": (provider == "Local Ollama")
        }
