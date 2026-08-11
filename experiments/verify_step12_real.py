import os
import sys
import json

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.services.model_registry import ModelRegistry

def run_step12_real_verification():
    print("=" * 80)
    print(" STEP 12 REAL MODEL REGISTRY VERIFICATION PASS (NO SELECTION / NO ROUTING LOGIC)")
    print("=" * 80)

    registry = ModelRegistry()
    models = registry.list_models()

    print(f"\nTotal Registered Models: {len(models)}\n")
    print("-" * 80)

    results = []

    for idx, m in enumerate(models, 1):
        print(f"Model [{idx}/{len(models)}]: {m.model_id}")
        print(f"  Display Name        : {m.display_name}")
        print(f"  Provider            : {m.provider}")
        print(f"  Type                : {m.model_type}")
        print(f"  Execution Mode      : {m.execution_mode}")
        print(f"  Local Execution     : {m.local}")
        print(f"  Available           : {m.available}")
        print(f"  Config Status       : {m.configuration_status.upper()}")
        print(f"  Capabilities        : {', '.join(m.capabilities)}")
        print(f"  Context Window      : {m.context_length} tokens" if m.context_length else "  Context Window      : N/A")
        print(f"  Embedding Dimension : {m.embedding_dimension}" if m.embedding_dimension else "  Embedding Dimension : N/A")
        print(f"  Requirements        : {m.requirements}")
        print(f"  Metadata Source     : {m.metadata_source}")
        print("-" * 80)

        results.append(m.dict())

    # Test individual lookup
    print("\nTesting Model Lookup:")
    bge_meta = registry.get_model("BAAI/bge-m3")
    print(f"  Lookup 'BAAI/bge-m3' -> Found: {bge_meta is not None}")
    if bge_meta:
        print(f"    Status: {bge_meta.configuration_status}, Available: {bge_meta.available}")

    unknown_meta = registry.get_model("nonexistent-model-xyz")
    print(f"  Lookup 'nonexistent-model-xyz' -> Found: {unknown_meta is not None} (Expected: False)")

    # Save verification report
    report_output_path = os.path.join(project_root, "data", "logs", "step12_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_count": len(models),
            "models": results
        }, f, indent=2)

    print(f"\nVerification report written to '{report_output_path}'.")
    print("=" * 80)

if __name__ == "__main__":
    run_step12_real_verification()
