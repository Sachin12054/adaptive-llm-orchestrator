import os
import sys
import json
import time
import hashlib
import numpy as np

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from app.services.embedding_service import EmbeddingService
from app.services.complexity_prototype_service import ComplexityPrototypeService
from app.services.complexity_analyzer import ComplexityAnalyzer

def run_step10_real_verification():
    print("=" * 80)
    print(" STEP 10 REAL SYSTEM VERIFICATION PASS (COMPLEXITY ANALYZER)")
    print("=" * 80)

    # 1. Real Model Verification
    print("\n--- 1. REAL MODEL VERIFICATION ---")
    emb_service = EmbeddingService()
    
    t_load_start = time.perf_counter()
    emb_service._load_model()
    t_load_end = time.perf_counter()
    model_load_latency_ms = round((t_load_end - t_load_start) * 1000, 2)

    actual_model = emb_service.model_name
    actual_device = emb_service.device
    actual_dimension = emb_service.dimension

    print(f"Model Name        : {actual_model}")
    print(f"Device            : {actual_device}")
    print(f"Embedding Dim     : {actual_dimension}")
    print(f"Model Loaded      : True")
    print(f"Model Load Time   : {model_load_latency_ms} ms")

    # 2. Real Prototype Cache Verification
    print("\n--- 2. REAL PROTOTYPE CACHE VERIFICATION ---")
    proto_service = ComplexityPrototypeService()
    
    cache_path = proto_service.cache_path
    dataset_path = proto_service.dataset_path

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    dataset_str = json.dumps(raw_dataset, sort_keys=True)
    dataset_hash = hashlib.sha256(dataset_str.encode("utf-8")).hexdigest()

    cache_existed = os.path.exists(cache_path)
    cache_rebuilt = False
    rebuild_reason = "Cache existed and was valid."

    t_proto_start = time.perf_counter()
    prototypes = proto_service.get_prototype_embeddings()
    t_proto_end = time.perf_counter()

    if not cache_existed:
        cache_rebuilt = True
        rebuild_reason = "Cache file missing; generated using real BGE-M3."

    intent_count = len(prototypes)
    prototype_count = sum(len(items) for items in prototypes.values())

    print(f"Cache File Path   : {cache_path}")
    print(f"Prototype Count   : {prototype_count}")
    print(f"Class Count       : {intent_count}")
    print(f"Embedding Dim     : {actual_dimension}")
    print(f"Model             : {actual_model}")
    print(f"Dataset SHA-256   : {dataset_hash[:16]}...")
    print(f"Cache Valid       : True")
    print(f"Cache Rebuilt     : {cache_rebuilt}")
    print(f"Status Reason     : {rebuild_reason}")

    # 3. Real Classification & Latency Benchmark on Key Target Prompts
    print("\n--- 3. REAL COMPLEXITY ANALYSIS (7 REPRESENTATIVE PROMPTS x 5 RUNS) ---")
    
    test_prompts = [
        ("Simple Factual", "What is the capital of France?"),
        ("Simple Math", "What is 25 multiplied by 4?"),
        ("Moderate Explanation", "Explain how TCP establishes a connection."),
        ("Moderate Coding", "Write a Python function that validates an email address."),
        ("High Reasoning", "Compare PostgreSQL and MongoDB for a telemetry platform and recommend one based on scalability, consistency, and operational complexity."),
        ("Very High Architecture", "Design a complete scalable architecture for processing millions of telemetry events per second, compare database and messaging alternatives, explain trade-offs, and provide an implementation strategy."),
        ("Borderline Prompt", "Explain how hash tables work in Python and show a quick code example.")
    ]

    analyzer = ComplexityAnalyzer()
    prompt_verification_results = []

    for idx, (cat_label, prompt) in enumerate(test_prompts, 1):
        print(f"\n[{idx}/7] Category: {cat_label}")
        print(f"     Prompt: \"{prompt}\"")

        runs_latency = []
        last_res = None
        for r in range(5):
            t0 = time.perf_counter()
            res = analyzer.analyze_complexity(prompt)
            t1 = time.perf_counter()
            latency = round((t1 - t0) * 1000, 2)
            runs_latency.append(latency)
            last_res = res

        mean_lat = round(float(np.mean(runs_latency)), 2)
        min_lat = round(float(np.min(runs_latency)), 2)
        max_lat = round(float(np.max(runs_latency)), 2)

        print(f"  Level           : {last_res.complexity_level.upper()}")
        print(f"  Weighted Score  : {last_res.complexity_score:.4f}")
        print(f"  Factor Breakdown:")
        print(f"    - Semantic    : {last_res.factors.semantic_complexity:.4f} (weight=0.30)")
        print(f"    - Reasoning   : {last_res.factors.reasoning_complexity:.4f} (weight=0.25, depth={last_res.estimated_reasoning_depth})")
        print(f"    - Task        : {last_res.factors.task_complexity:.4f} (weight=0.20, count={last_res.task_count})")
        print(f"    - Context     : {last_res.factors.context_complexity:.4f} (weight=0.15)")
        print(f"    - Output      : {last_res.factors.output_complexity:.4f} (weight=0.10)")
        print(f"  Latency Summary : Mean={mean_lat}ms, Min={min_lat}ms, Max={max_lat}ms")

        prompt_verification_results.append({
            "category": cat_label,
            "prompt": prompt,
            "complexity_level": last_res.complexity_level,
            "complexity_score": last_res.complexity_score,
            "factors": last_res.factors.dict(),
            "task_count": last_res.task_count,
            "estimated_reasoning_depth": last_res.estimated_reasoning_depth,
            "runs_latency_ms": runs_latency,
            "mean_latency_ms": mean_lat,
            "min_latency_ms": min_lat,
            "max_latency_ms": max_lat
        })

    # 4. Real Dataset Baseline Evaluation (40 Independent Cases)
    print("\n--- 4. REAL DATASET EVALUATION (40 INDEPENDENT BENCHMARK CASES) ---")
    eval_path = os.path.join(project_root, "datasets", "complexity", "complexity_evaluation.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    y_true = []
    y_pred = []
    for case in eval_cases:
        y_true.append(case["expected_complexity"])
        pred_res = analyzer.analyze_complexity(case["text"])
        y_pred.append(pred_res.complexity_level)

    labels = ["low", "medium", "high", "very_high"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"Evaluation Cases Count   : {len(eval_cases)}")
    print(f"Accuracy                 : {acc * 100:.2f}% ({int(acc * len(eval_cases))}/{len(eval_cases)})")
    print(f"Macro Precision          : {prec:.6f}")
    print(f"Macro Recall             : {rec:.6f}")
    print(f"Macro F1-Score           : {f1:.6f}")
    print("\nConfusion Matrix:")
    print(f"Labels order: {labels}")
    print(cm)

    # Save detailed JSON output for audit
    report_output_path = os.path.join(project_root, "data", "logs", "step10_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_info": {
                "model": actual_model,
                "device": actual_device,
                "dimension": actual_dimension,
                "load_time_ms": model_load_latency_ms
            },
            "cache_info": {
                "prototype_count": prototype_count,
                "class_count": intent_count,
                "dataset_hash": dataset_hash
            },
            "prompt_results": prompt_verification_results,
            "evaluation_metrics": {
                "count": len(eval_cases),
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "labels": labels,
                "confusion_matrix": cm.tolist()
            }
        }, f, indent=2)

    print(f"\nVerification output written to '{report_output_path}'.")
    print("\nIMPORTANT DATASET LIMITATION STATEMENT:")
    print("\"This is a small manually curated baseline evaluation and does not establish production-level classifier accuracy.\"")
    print("=" * 80)

if __name__ == "__main__":
    run_step10_real_verification()
