import os
import sys
import time
from fastapi.testclient import TestClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app
from app.services.orchestration_pipeline import OrchestrationPipeline
from app.schemas.orchestration import OrchestrationRequest

client = TestClient(app)

def run_benchmarks():
    print("=" * 80)
    print(" THREE-PROMPT E2E LATENCY & TELEMETRY BENCHMARK")
    print("=" * 80)

    prompts = [
        ("SIMPLE", "What is Python?"),
        ("SIMPLE CODING", "Write Python code to read a CSV using pandas."),
        ("MEDIUM", "Write a Python script that reads a CSV, cleans missing values, and trains a linear regression model.")
    ]

    pipeline = OrchestrationPipeline()
    results = []

    # Warmup call
    pipeline.run_pipeline(OrchestrationRequest(prompt="Hello"))

    for label, text in prompts:
        print(f"\n--- Benchmark [{label}]: \"{text}\" ---")
        t0 = time.perf_counter()
        res = pipeline.run_pipeline(OrchestrationRequest(prompt=text))
        t1 = time.perf_counter()

        actual_total_latency_ms = round((t1 - t0) * 1000, 2)
        gen = res.generation
        ver = res.verification
        rew = res.reward

        record = {
            "label": label,
            "prompt": text,
            "total_latency_ms": actual_total_latency_ms,
            "pipeline_reported_ms": res.pipeline_latency_ms,
            "selected_model": res.selected_model,
            "generation_latency_ms": gen.latency_ms if gen else 0,
            "ollama_load_ms": getattr(gen, "ollama_load_ms", 0) or 0,
            "ollama_prompt_eval_ms": getattr(gen, "ollama_prompt_eval_ms", 0) or 0,
            "ollama_eval_ms": getattr(gen, "ollama_eval_ms", 0) or 0,
            "prompt_tokens": gen.usage.input_tokens if (gen and gen.usage) else "N/A",
            "output_tokens": gen.usage.output_tokens if (gen and gen.usage) else "N/A",
            "verification_latency_ms": ver.verification_latency_ms if ver else 0,
            "reward": rew.reward if rew else 0.0
        }
        results.append(record)

        print(f"  Total Latency         : {actual_total_latency_ms} ms (Warm Model Pipeline)")
        print(f"  Selected Model        : {record['selected_model']}")
        print(f"  Generation Latency    : {record['generation_latency_ms']} ms")
        print(f"  Ollama Load Latency   : {record['ollama_load_ms']} ms")
        print(f"  Ollama Eval Latency   : {record['ollama_eval_ms']} ms")
        print(f"  Prompt / Output Tokens: {record['prompt_tokens']} / {record['output_tokens']}")
        print(f"  Verification Latency  : {record['verification_latency_ms']} ms")
        print(f"  Computed Reward       : {record['reward']:.4f}")
        print(f"  Generated Text Output :\n{str(gen.generated_text)[:120]}...\n")

    print("=" * 80)
    print(" SUMMARY BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Category':<15} | {'Model':<15} | {'Total Latency':<14} | {'Gen Latency':<12} | {'Tokens (In/Out)':<15} | {'Reward':<7}")
    print("-" * 86)
    for r in results:
        toks = f"{r['prompt_tokens']}/{r['output_tokens']}"
        print(f"{r['label']:<15} | {r['selected_model']:<15} | {r['total_latency_ms']:>8} ms     | {r['generation_latency_ms']:>6} ms     | {toks:<15} | {r['reward']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()
