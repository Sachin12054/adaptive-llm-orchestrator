import logging
from typing import Dict, Any

logger = logging.getLogger("validation")

class FixedModelPolicy:
    """
    Fixed Model Routing Strategy (A, B, or C).
    Always selects a specified single model regardless of task context.
    """

    def __init__(self, model_id: str, provider: str = "Local Ollama"):
        self.model_id = model_id
        self.provider = provider

    def decide(self, prompt: str, category: str, execution_mode: str = "online") -> Dict[str, Any]:
        return {
            "selected_model": self.model_id,
            "provider": self.provider,
            "decision_latency_ms": 0.5,
            "strategy": f"fixed_{self.model_id}"
        }
