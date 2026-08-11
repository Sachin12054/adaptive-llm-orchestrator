import sys
import pytest

if __name__ == "__main__":
    test_files = [
        "backend/tests/test_adaptive_decision_engine.py",
        "backend/tests/test_response_generator.py",
        "backend/tests/test_response_verifier.py",
        "backend/tests/test_reward_signal.py",
        "backend/tests/test_orchestration_pipeline.py"
    ]
    print(f"Running pytest on: {test_files}")
    ret = pytest.main(["-v"] + test_files)
    sys.exit(ret)
