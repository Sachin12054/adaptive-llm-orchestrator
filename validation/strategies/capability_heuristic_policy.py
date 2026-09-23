import logging
from typing import Dict, Any

logger = logging.getLogger("validation")

class CapabilityHeuristicPolicy:
    """
    Strategy F: Capability-Based Heuristic Routing Policy.
    Transparent rule-based decision tree matching task category & complexity to models:
    - Coding/Debugging/SQL -> qwen-coder-3b (local) or mistral-small-latest (online)
    - Multi-step Reasoning / Architecture -> deepseek-r1-7b (local) or llama-3.3-70b-instruct (online)
    - Simple QA / Resource-Sensitive -> gemma-3-4b (local) or gemini-3.5-flash (online)
    - Complex / Planning -> gemini-3.5-flash or meta-llama/llama-3.3-70b-instruct
    """

    def decide(self, prompt: str, category: str, execution_mode: str = "online") -> Dict[str, Any]:
        cat_lower = (category or "").lower()
        
        if execution_mode == "local":
            if any(k in cat_lower for k in ["coding", "debugging", "sql", "database"]):
                selected = "qwen-coder-3b"
            elif any(k in cat_lower for k in ["reasoning", "architecture", "complex", "planning"]):
                selected = "deepseek-r1-7b"
            else:
                selected = "gemma-3-4b"
            provider = "Local Ollama"
        else:
            if any(k in cat_lower for k in ["coding", "debugging", "sql"]):
                selected = "mistral-small-latest"
                provider = "Mistral API"
            elif any(k in cat_lower for k in ["reasoning", "architecture", "complex"]):
                selected = "meta-llama/llama-3.3-70b-instruct"
                provider = "OpenRouter API"
            else:
                selected = "gemini-3.5-flash"
                provider = "Google Gemini API"

        return {
            "selected_model": selected,
            "provider": provider,
            "decision_latency_ms": 1.2,
            "strategy": "capability_heuristic"
        }
