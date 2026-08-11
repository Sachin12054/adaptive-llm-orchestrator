import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from experiments.run_verification_suite import main

if __name__ == "__main__":
    main()
