import sys
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer

def main():
    prompt = "Design a production-ready distributed machine learning pipeline for real-time fraud detection using Python. Break the architecture into data ingestion, preprocessing, feature engineering, model training, model serving, monitoring, fault tolerance, scalability, and deployment. Explain how each component communicates with the others and provide implementation examples."

    decomposer = ComplexTaskDecomposer()
    plan = decomposer.decompose(prompt)

    print("=== Complex Task Decomposer Response ===")
    print(f"is_complex: {plan.is_complex}")
    print(f"total_subtasks: {plan.total_subtasks}")
    print(f"plan_latency_ms: {plan.plan_latency_ms} ms")
    print(f"execution_levels: {plan.execution_levels}")
    print("\nSubtasks generated:")
    for st in plan.subtasks:
        print(f" - [{st.task_id}] ({st.category}): {st.description[:60]}... -> Model: {st.assigned_model}")

    assert len(plan.subtasks) > 0, "Subtasks list must not be empty"
    print("\nVERIFICATION SUCCESS: Backend returns actual subtask array for complex prompt!")

if __name__ == "__main__":
    main()
