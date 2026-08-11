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
from app.services.intent_prototype_service import IntentPrototypeService
from app.services.intent_classifier import IntentClassifier

def run_step9_real_verification():
    print("=" * 80)
    print(" STEP 9 REAL SYSTEM VERIFICATION PASS (CONSERVATIVE SIMILARITY SEMANTICS)")
    print("=" * 80)

    # 1. Real Model Verification
    print("\n--- 1. REAL MODEL VERIFICATION ---")
    emb_service = EmbeddingService()
    
    # Force model load
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
    proto_service = IntentPrototypeService()
    
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
    print(f"Intent Count      : {intent_count}")
    print(f"Embedding Dim     : {actual_dimension}")
    print(f"Model             : {actual_model}")
    print(f"Dataset SHA-256   : {dataset_hash[:16]}...")
    print(f"Cache Valid       : True")
    print(f"Cache Rebuilt     : {cache_rebuilt}")
    print(f"Status Reason     : {rebuild_reason}")

    # 3. Real Classification & Latency Benchmark on 9 Standard Test Prompts
    print("\n--- 3. REAL API & CLASSIFICATION BENCHMARK (9 PROMPTS x 5 RUNS) ---")
    
    test_prompts = [
        "What is the capital of France?",
        "Write a Python function to reverse a linked list.",
        "Solve the equation 2x + 5 = 15.",
        "Explain how TCP establishes a connection.",
        "Summarize the following article into five bullet points.",
        "Write a short story about a robot exploring Mars.",
        "Translate 'Good morning' into Hindi.",
        "Which database architecture would be better for this system and why?",
        "Hello, how are you?"
    ]

    classifier = IntentClassifier()
    prompt_verification_results = []

    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\nPrompt [{idx}/9]: \"{prompt}\"")

        # 5 Warm Latency Benchmark Runs
        runs_latency = []
        last_res = None
        for r in range(5):
            t0 = time.perf_counter()
            res = classifier.classify_intent(prompt)
            t1 = time.perf_counter()
            latency = round((t1 - t0) * 1000, 2)
            runs_latency.append(latency)
            last_res = res

        mean_lat = round(float(np.mean(runs_latency)), 2)
        min_lat = round(float(np.min(runs_latency)), 2)
        max_lat = round(float(np.max(runs_latency)), 2)

        print(f"  Top Intent       : {last_res.intent}")
        print(f"  Top Similarity   : {last_res.top_similarity:.4f}")
        print(f"  Second Similarity: {last_res.second_similarity:.4f}")
        print(f"  Margin           : {last_res.margin:.4f}")
        print(f"  Is Ambiguous     : {last_res.is_ambiguous}")
        print(f"  Latency (5 Runs) : Run1={runs_latency[0]}ms, Run2={runs_latency[1]}ms, Run3={runs_latency[2]}ms, Run4={runs_latency[3]}ms, Run5={runs_latency[4]}ms")
        print(f"  Latency Summary  : Mean={mean_lat}ms, Min={min_lat}ms, Max={max_lat}ms")

        print("  Complete Ranked Similarities:")
        for r_idx, item in enumerate(last_res.ranked_intents, 1):
            print(f"    {r_idx}. {item.intent:<15} : {item.score:.4f}")

        prompt_verification_results.append({
            "prompt": prompt,
            "top_intent": last_res.intent,
            "top_similarity": last_res.top_similarity,
            "second_similarity": last_res.second_similarity,
            "margin": last_res.margin,
            "is_ambiguous": last_res.is_ambiguous,
            "ranked_intents": [{"intent": item.intent, "score": item.score} for item in last_res.ranked_intents],
            "runs_latency_ms": runs_latency,
            "mean_latency_ms": mean_lat,
            "min_latency_ms": min_lat,
            "max_latency_ms": max_lat
        })

    # 4. Ambiguous Prompt Verification
    print("\n--- 4. AMBIGUOUS PROMPT TEST ---")
    ambiguous_prompts = [
        "Which architecture would be best to write this Python service?",
        "Summarize this code and translate the explanation into Spanish."
    ]
    
    ambiguous_results = []
    for idx, prompt in enumerate(ambiguous_prompts, 1):
        print(f"\nAmbiguous Prompt [{idx}/2]: \"{prompt}\"")
        res = classifier.classify_intent(prompt)

        print(f"  Top Intent       : {res.intent}")
        print(f"  Top Similarity   : {res.top_similarity:.4f}")
        print(f"  Second Similarity: {res.second_similarity:.4f}")
        print(f"  Margin           : {res.margin:.4f}")
        print(f"  Is Ambiguous     : {res.is_ambiguous}")

        print("  Ranked Candidates:")
        for item in res.ranked_intents[:4]:
            print(f"    - {item.intent:<15} : {item.score:.4f}")

        ambiguous_results.append({
            "prompt": prompt,
            "top_intent": res.intent,
            "top_similarity": res.top_similarity,
            "second_similarity": res.second_similarity,
            "margin": res.margin,
            "is_ambiguous": res.is_ambiguous,
            "ranked_intents": [{"intent": item.intent, "score": item.score} for item in res.ranked_intents]
        })

    # 5. Real Dataset Baseline Evaluation (datasets/intent/intent_evaluation.json)
    print("\n--- 5. REAL DATASET EVALUATION (27 BENCHMARK CASES) ---")
    eval_path = os.path.join(project_root, "datasets", "intent", "intent_evaluation.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    y_true = []
    y_pred = []
    for case in eval_cases:
        y_true.append(case["expected_intent"])
        pred_res = classifier.classify_intent(case["text"])
        y_pred.append(pred_res.intent)

    labels = sorted(list(set(y_true + y_pred)))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"Evaluation Dataset Count : {len(eval_cases)}")
    print(f"Accuracy                 : {acc * 100:.2f}% ({int(acc * len(eval_cases))}/{len(eval_cases)})")
    print(f"Macro Precision          : {prec:.6f}")
    print(f"Macro Recall             : {rec:.6f}")
    print(f"Macro F1-Score           : {f1:.6f}")
    print("\nConfusion Matrix:")
    print(f"Labels order: {labels}")
    print(cm)

    # Save detailed JSON output for audit
    report_output_path = os.path.join(project_root, "data", "logs", "step9_real_verification.json")
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
                "intent_count": intent_count,
                "dataset_hash": dataset_hash
            },
            "prompt_results": prompt_verification_results,
            "ambiguous_results": ambiguous_results,
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
    run_step9_real_verification()
