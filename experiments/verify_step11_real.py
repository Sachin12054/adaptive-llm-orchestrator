import os
import sys
import json
import time

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from app.services.resource_analyzer import ResourceAnalyzer

def run_step11_real_verification():
    print("=" * 80)
    print(" STEP 11 REAL SYSTEM RESOURCE TELEMETRY VERIFICATION PASS (NO FAKE DATA)")
    print("=" * 80)

    analyzer = ResourceAnalyzer()

    t0 = time.perf_counter()
    snapshot = analyzer.get_resource_snapshot()
    t1 = time.perf_counter()

    exec_latency_ms = round((t1 - t0) * 1000, 2)

    print("\n--- 1. CPU TELEMETRY ---")
    print(f"Physical Cores      : {snapshot.cpu.physical_cores}")
    print(f"Logical Cores       : {snapshot.cpu.logical_cores}")
    print(f"Utilization         : {snapshot.cpu.utilization_percent}%")
    print(f"Frequency           : {snapshot.cpu.frequency_mhz} MHz" if snapshot.cpu.frequency_mhz else "Frequency : N/A")

    print("\n--- 2. MEMORY (RAM) TELEMETRY ---")
    print(f"Total RAM           : {snapshot.memory.total_gb} GB")
    print(f"Available RAM       : {snapshot.memory.available_gb} GB")
    print(f"Used RAM            : {snapshot.memory.used_gb} GB")
    print(f"Utilization         : {snapshot.memory.utilization_percent}%")

    print("\n--- 3. GPU & CUDA TELEMETRY ---")
    print(f"CUDA Usable in PyTorch: {snapshot.cuda.available}")
    print(f"PyTorch Version       : {snapshot.cuda.pytorch_version}")
    print(f"CUDA Version          : {snapshot.cuda.cuda_version}")
    print(f"GPU Available         : {snapshot.gpu.available}")
    print(f"GPU Name              : {snapshot.gpu.name}")
    print(f"Device Count          : {snapshot.gpu.device_count}")
    print(f"Total VRAM            : {snapshot.gpu.total_vram_gb} GB" if snapshot.gpu.total_vram_gb else "Total VRAM : N/A")
    print(f"Free VRAM             : {snapshot.gpu.free_vram_gb} GB" if snapshot.gpu.free_vram_gb else "Free VRAM : N/A")

    print("\n--- 4. APPLICATION PROCESS TELEMETRY ---")
    print(f"Process PID         : {snapshot.process.pid}")
    print(f"Memory RSS          : {snapshot.process.memory_mb} MB")
    print(f"Process CPU %       : {snapshot.process.cpu_percent}%")

    print("\n--- 5. DISK STORAGE TELEMETRY ---")
    print(f"Total Storage       : {snapshot.disk.total_gb} GB")
    print(f"Free Storage        : {snapshot.disk.free_gb} GB")
    print(f"Used Storage        : {snapshot.disk.used_gb} GB")
    print(f"Utilization         : {snapshot.disk.utilization_percent}%")

    print("\n--- 6. RUNTIME ENVIRONMENT ---")
    print(f"Operating System    : {snapshot.runtime.operating_system}")
    print(f"Python Version      : {snapshot.runtime.python_version}")
    print(f"Embedding Device    : {snapshot.runtime.execution_device}")

    print("\n--- 7. RESOURCE FEASIBILITY ANALYSIS ---")
    print(f"Status              : {snapshot.feasibility.status.upper()}")
    print(f"Rationale           : {snapshot.feasibility.notes}")

    print("\n--- 8. TELEMETRY MEASUREMENT PERFORMANCE ---")
    print(f"Snapshot Latency    : {snapshot.measurement_latency_ms} ms (Wrapper Execution: {exec_latency_ms} ms)")

    # Save detailed JSON log
    report_output_path = os.path.join(project_root, "data", "logs", "step11_real_verification.json")
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot.dict(), f, indent=2)

    print(f"\nReal resource snapshot written to '{report_output_path}'.")
    print("=" * 80)

if __name__ == "__main__":
    run_step11_real_verification()
