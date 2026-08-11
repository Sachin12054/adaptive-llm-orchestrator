import sys
import os
import json
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
sys.path.insert(0, backend_dir)

from experiments.verify_all_model_identities import run_identity_audit

if __name__ == "__main__":
    print("Executing full model identity verification suite...")
    asyncio.run(run_identity_audit())
