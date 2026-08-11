import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from experiments.test_fastapi_sse_startup import main as run_fastapi_sse_test

if __name__ == "__main__":
    run_fastapi_sse_test()
