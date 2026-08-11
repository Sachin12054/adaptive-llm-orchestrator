import logging
from typing import Dict, List, Optional

logger = logging.getLogger("orchestrator")

class TaskAllocator:
    """
    Model Allocation Layer.
    Refers strictly to BaselineAdaptivePolicy for model selection.
    DOES NOT independently override or select candidate models.
    """

    def allocate_model(
        self,
        category: str,
        execution_mode: str = "local",
        policy_selected_model: Optional[str] = None
    ) -> str:
        if not policy_selected_model or not policy_selected_model.strip():
            logger.error(f"[TASK ALLOCATION ERROR] Subtask category '{category}' allocation failed: No authoritative policy model provided from BaselineAdaptivePolicy.")
            raise ValueError(f"TaskAllocator requires authoritative policy_selected_model from BaselineAdaptivePolicy for subtask category '{category}'.")

        selected = policy_selected_model.strip()
        logger.info(f"[TASK ALLOCATION] Subtask category '{category}' assigned authoritative policy model '{selected}'.")
        return selected
