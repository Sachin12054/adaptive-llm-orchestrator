import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.schemas.model import ModelMetadata
from app.services.policies.baseline_policy import BaselineAdaptivePolicy

def test_scoring():
    policy = BaselineAdaptivePolicy()

    gemma_meta = ModelMetadata(
        model_id="gemma-3-4b", provider="ollama", display_name="Gemma 3 4B", model_type="llm",
        capabilities=["general_qa", "explanation", "conversational", "summarization", "factual", "creative"],
        context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"
    )
    coder_meta = ModelMetadata(
        model_id="qwen-coder-3b", provider="ollama", display_name="Qwen Coder 3B", model_type="llm",
        capabilities=["coding", "explanation", "general_qa"],
        context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"
    )
    deepseek_meta = ModelMetadata(
        model_id="deepseek-r1-7b", provider="ollama", display_name="DeepSeek R1 7B", model_type="llm",
        capabilities=["reasoning", "mathematics", "coding", "general_qa"],
        context_length=8192, execution_mode="local", local=True, available=True, configuration_status="configured"
    )

    intent_info = {"intent": "coding", "is_ambiguous": True}
    complexity_info = {"level": "medium", "complexity_score": 0.5661}
    resource_info = {"gpu_available": True, "gpu_free_vram_gb": 3.5, "memory_utilization_percent": 50.0, "cpu_utilization_percent": 20.0}

    selected_model, winning_score, breakdowns, reasoning = policy.evaluate_candidates(
        prompt="Design a distributed Python ML pipeline...",
        intent_info=intent_info,
        complexity_info=complexity_info,
        resource_info=resource_info,
        candidate_models=[gemma_meta, coder_meta, deepseek_meta]
    )

    print("Current Baseline Policy Evaluation Result:")
    print(f"Selected Model: {selected_model} (Winning Score: {winning_score:.4f})")
    for b in breakdowns:
        print(f"  {b.model_id:<15}: eligible={b.eligible}, cap={b.capability_score:.2f}, cmplx={b.complexity_fit_score:.2f}, res={b.resource_fit_score:.2f}, ctx={b.context_fit_score:.2f} -> total={b.candidate_score:.4f}")

if __name__ == "__main__":
    test_scoring()
