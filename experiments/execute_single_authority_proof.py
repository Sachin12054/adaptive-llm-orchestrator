import sys
import os
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from experiments.run_final_single_authority_proof import run_final_single_authority_proof

if __name__ == "__main__":
    asyncio.run(run_final_single_authority_proof())
