import logging
from typing import Dict, Any, List

logger = logging.getLogger("validation")

class RoundRobinPolicy:
    """
    Strategy E: Round-Robin Routing.
    Cyclic rotation through active eligible models.
    """

    def __init__(self, eligible_models: List[str] = None):
        self.eligible_models = eligible_models or [
            "gemma-3-4b",
            "qwen-coder-3b",
            "deepseek-r1-7b",
            "gemini-3.5-flash",
            "mistral-small-latest",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-3.3-70b-instruct"
        ]
        self.counter = 0

    def decide(self, prompt: str, category: str, execution_mode: str = "online") -> Dict[str, Any]:
        models = [m for m in self.eligible_models if m != "BAAI/bge-m3"]
        if execution_mode == "local":
            models = [m for m in models if m in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]]

        selected = models[self.counter % len(models)]
        self.counter += 1
        provider = "Local Ollama" if selected in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] else "Cloud API"

        return {
            "selected_model": selected,
            "provider": provider,
            "decision_latency_ms": 0.5,
            "strategy": "round_robin"
        }
