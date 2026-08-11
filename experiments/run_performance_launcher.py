import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.verify_performance_and_ollama_telemetry import main as run_telemetry

if __name__ == "__main__":
    run_telemetry()
