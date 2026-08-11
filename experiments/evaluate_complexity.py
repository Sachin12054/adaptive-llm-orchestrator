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

from app.services.complexity_analyzer import ComplexityAnalyzer

def run_complexity_evaluation():
    eval_dataset_path = os.path.join(project_root, "datasets", "complexity", "complexity_evaluation.json")
    if not os.path.exists(eval_dataset_path):
        print(f"Error: Complexity evaluation dataset not found at {eval_dataset_path}")
        sys.exit(1)

    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Loaded {len(eval_data)} complexity evaluation test cases from '{eval_dataset_path}'.")

    analyzer = ComplexityAnalyzer()
    
    y_true = []
    y_pred = []
    scores_by_class = {"low": [], "medium": [], "high": [], "very_high": []}
    latencies = []
    results_detail = []

    print("\nExecuting Complexity Analyzer Evaluation using REAL BGE-M3 model...\n")

    for idx, item in enumerate(eval_data, 1):
        prompt = item["text"]
        expected = item["expected_complexity"]

        t0 = time.perf_counter()
        res = analyzer.analyze_complexity(prompt)
        t1 = time.perf_counter()

        latency_ms = round((t1 - t0) * 1000, 2)
        latencies.append(latency_ms)

        predicted = res.complexity_level
        y_true.append(expected)
        y_pred.append(predicted)

        scores_by_class[expected].append(res.complexity_score)

        is_correct = (expected == predicted)
        status_mark = "✓" if is_correct else "✗"

        print(f"[{idx:02d}/{len(eval_data):02d}] {status_mark} Expected: {expected:<10} | Predicted: {predicted:<10} | Score: {res.complexity_score:.4f} | Latency: {latency_ms:.1f}ms")
        print(f"     Prompt: '{prompt[:75]}...'")

        results_detail.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "complexity_score": res.complexity_score,
            "factors": res.factors.dict(),
            "task_count": res.task_count,
            "estimated_reasoning_depth": res.estimated_reasoning_depth,
            "latency_ms": latency_ms
        })

    # Metrics computation
    labels = ["low", "medium", "high", "very_high"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    avg_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0

    mean_scores = {
        cls: float(sum(s) / len(s)) if s else 0.0
        for cls, s in scores_by_class.items()
    }

    print("\n" + "=" * 65)
    print("COMPLEXITY ANALYZER EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Total Samples Evaluated : {len(eval_data)}")
    print(f"Overall Accuracy        : {acc * 100:.2f}% ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(eval_data)})")
    print(f"Macro Precision        : {prec:.4f}")
    print(f"Macro Recall           : {rec:.4f}")
    print(f"Macro F1-Score         : {f1:.4f}")
    print(f"Average Latency        : {avg_latency:.2f} ms")
    print("-" * 65)
    print("Mean Weighted Complexity Score by Expected Class:")
    for cls in labels:
        print(f"  - {cls:<10} : {mean_scores[cls]:.4f}")
    print("=" * 65)

    print("\nPer-Class Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("Confusion Matrix:")
    print(f"Labels: {labels}\n")
    print(cm)
    print("\n" + "=" * 65)

    eval_output_path = os.path.join(project_root, "data", "logs", "complexity_evaluation_results.json")
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "avg_latency_ms": avg_latency,
                "mean_scores_by_class": mean_scores
            },
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "details": results_detail
        }, f, indent=2)

    print(f"Detailed complexity evaluation metrics saved to '{eval_output_path}'.")

if __name__ == "__main__":
    run_complexity_evaluation()
