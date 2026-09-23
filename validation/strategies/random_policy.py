import random
import logging
from typing import Dict, Any, List

logger = logging.getLogger("validation")

class RandomPolicy:
    """
    Strategy D: Random Routing.
    Uniform random selection among eligible generation models using a reproducible seed.
    """

    def __init__(self, seed: int = 42, eligible_models: List[str] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.eligible_models = eligible_models or [
            "gemma-3-4b",
            "qwen-coder-3b",
            "deepseek-r1-7b",
            "gemini-3.5-flash",
            "mistral-small-latest",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-3.3-70b-instruct"
        ]

    def decide(self, prompt: str, category: str, execution_mode: str = "online") -> Dict[str, Any]:
        # Exclude BAAI/bge-m3 (Action 7 masked)
        models = [m for m in self.eligible_models if m != "BAAI/bge-m3"]
        if execution_mode == "local":
            models = [m for m in models if m in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]]

        selected = self.rng.choice(models)
        provider = "Local Ollama" if selected in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"] else "Cloud API"

        return {
            "selected_model": selected,
            "provider": provider,
            "decision_latency_ms": 0.8,
            "strategy": "random"
        }
