import sys
import os
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "backend"))

from app.services.complex.task_allocator import TaskAllocator

def test_task_allocator_requires_policy_model():
    allocator = TaskAllocator()

    print("\n==================================================")
    print("  TESTING TASK ALLOCATOR WITHOUT POLICY MODEL")
    print("==================================================")

    # 1. Test None policy model raises ValueError
    with pytest.raises(ValueError) as exc_info_1:
        allocator.allocate_model("coding", execution_mode="online", policy_selected_model=None)
    print(f" -> Successfully caught expected ValueError when policy_selected_model=None:\n    '{exc_info_1.value}'")

    # 2. Test empty string policy model raises ValueError
    with pytest.raises(ValueError) as exc_info_2:
        allocator.allocate_model("reasoning", execution_mode="local", policy_selected_model="   ")
    print(f" -> Successfully caught expected ValueError when policy_selected_model='  ':\n    '{exc_info_2.value}'")

    # 3. Test valid policy model returns exact model without defaulting
    model = allocator.allocate_model("coding", execution_mode="online", policy_selected_model="gemini-2.5-flash")
    assert model == "gemini-2.5-flash"
    print(f" -> Valid policy model passed through correctly: '{model}'")

    print("\n=> TASK ALLOCATION FAILURE & NO-FALLBACK TEST PASSED 100%!")

if __name__ == "__main__":
    test_task_allocator_requires_policy_model()
