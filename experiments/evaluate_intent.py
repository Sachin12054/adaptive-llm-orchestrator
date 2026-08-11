import os
import sys
import json
import time

# Ensure backend package is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from app.services.intent_classifier import IntentClassifier

def run_intent_evaluation():
    eval_dataset_path = os.path.join(project_root, "datasets", "intent", "intent_evaluation.json")
    if not os.path.exists(eval_dataset_path):
        print(f"Error: Evaluation dataset not found at {eval_dataset_path}")
        sys.exit(1)

    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Loaded {len(eval_data)} evaluation test cases from '{eval_dataset_path}'.")

    classifier = IntentClassifier()
    
    y_true = []
    y_pred = []
    latencies = []
    results_detail = []

    print("\nExecuting Intent Classification Evaluation using REAL BGE-M3 model...\n")

    for idx, item in enumerate(eval_data, 1):
        prompt = item["text"]
        expected = item["expected_intent"]

        t0 = time.perf_counter()
        res = classifier.classify_intent(prompt)
        t1 = time.perf_counter()

        latency_ms = round((t1 - t0) * 1000, 2)
        latencies.append(latency_ms)

        predicted = res.intent
        y_true.append(expected)
        y_pred.append(predicted)

        is_correct = (expected == predicted)
        status_mark = "✓" if is_correct else "✗"

        print(f"[{idx:02d}/{len(eval_data):02d}] {status_mark} Expected: {expected:<15} | Predicted: {predicted:<15} | TopSim: {res.top_similarity:.4f} | Margin: {res.margin:.4f} | Latency: {latency_ms:.1f}ms")
        print(f"     Prompt: '{prompt}'")

        results_detail.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "top_similarity": res.top_similarity,
            "second_similarity": res.second_similarity,
            "margin": res.margin,
            "is_ambiguous": res.is_ambiguous,
            "latency_ms": latency_ms
        })

    # Metrics computation
    labels = sorted(list(set(y_true + y_pred)))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    avg_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0

    print("\n" + "=" * 60)
    print("INTENT CLASSIFIER EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Samples Evaluated : {len(eval_data)}")
    print(f"Overall Accuracy        : {acc * 100:.2f}%")
    print(f"Macro Precision        : {prec:.4f}")
    print(f"Macro Recall           : {rec:.4f}")
    print(f"Macro F1-Score         : {f1:.4f}")
    print(f"Average Latency        : {avg_latency:.2f} ms")
    print("=" * 60)

    print("\nPer-Class Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("Confusion Matrix:")
    print(f"Labels: {labels}\n")
    print(cm)
    print("\n" + "=" * 60)

    eval_output_path = os.path.join(project_root, "data", "logs", "intent_evaluation_results.json")
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "avg_latency_ms": avg_latency
            },
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "details": results_detail
        }, f, indent=2)

    print(f"Detailed evaluation metrics saved to '{eval_output_path}'.")

if __name__ == "__main__":
    run_intent_evaluation()
