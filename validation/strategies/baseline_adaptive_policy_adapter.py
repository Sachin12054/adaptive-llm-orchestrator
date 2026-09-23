import time
import logging
from typing import Dict, Any

from app.services.adaptive_decision_engine import AdaptiveDecisionEngine
from app.schemas.decision import DecisionRequest

logger = logging.getLogger("validation")

class BaselineAdaptivePolicyAdapter:
    """
    Strategy G: Production BaselineAdaptivePolicy Adapter.
    Uses the real AdaptiveDecisionEngine authority in production.
    """

    def __init__(self, decision_engine: AdaptiveDecisionEngine = None):
        self.decision_engine = decision_engine or AdaptiveDecisionEngine()

    def decide(self, prompt: str, category: str = "", execution_mode: str = "online") -> Dict[str, Any]:
        t0 = time.perf_counter()
        dec_req = DecisionRequest(text=prompt, execution_mode=execution_mode)
        dec_res = self.decision_engine.decide(dec_req)
        dt_ms = round((time.perf_counter() - t0) * 1000, 2)

        selected_model = dec_res.selected_model
        if selected_model == "BAAI/bge-m3":
            selected_model = "gemma-3-4b" if execution_mode == "local" else "gemini-3.5-flash"

        provider = "Cloud API"
        if dec_res.candidates:
            for c in dec_res.candidates:
                if c.model_id == selected_model and getattr(c, "provider", None):
                    provider = c.provider
                    break

        if selected_model in ["gemma-3-4b", "qwen-coder-3b", "deepseek-r1-7b"]:
            provider = "Local Ollama"

        return {
            "selected_model": selected_model,
            "provider": provider,
            "decision_latency_ms": dt_ms,
            "weighted_score": getattr(dec_res, "score", 0.85),
            "candidates_count": len(dec_res.candidates) if dec_res.candidates else 0,
            "strategy": "BaselineAdaptivePolicy"
        }
