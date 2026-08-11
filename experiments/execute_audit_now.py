import sys
import os
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from experiments.run_real_e2e_audit import run_e2e_audit

if __name__ == "__main__":
    asyncio.run(run_e2e_audit())
