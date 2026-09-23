import pytest
from app.schemas.complex import SubTask
from app.services.complex.synthesis_strategies import (
    deterministic_s2_assembly,
    hybrid_s3_assembly,
    llm_synthesis_s0_s1
)

def test_deterministic_s2_assembly():
    subtasks = [
        SubTask(
            task_id="TASK-1",
            category="coding",
            task_objective="Design Database Schema",
            description="Create PostgreSQL tables for students and courses.",
            assigned_model="qwen-coder-3b",
            generated_text="```sql\nCREATE TABLE students (id INT PRIMARY KEY, name VARCHAR(100));\n```",
            execution_success=True
        ),
        SubTask(
            task_id="TASK-2",
            category="explanation",
            task_objective="Design System Architecture",
            description="Outline microservices and REST API gateways.",
            assigned_model="gemma-3-4b",
            generated_text="### Microservices Architecture\n- Auth Service\n- Navigation Service",
            execution_success=True
        )
    ]

    prompt = "Design a college assistant system"
    text, latency_ms = deterministic_s2_assembly(prompt, subtasks)

    assert "Executive Synthesis & Modular Analysis" in text
    assert "Design Database Schema" in text
    assert "CREATE TABLE students" in text
    assert "Microservices Architecture" in text
    assert latency_ms >= 0.0

    # Reproducibility check: second call must be byte-for-byte identical
    text2, _ = deterministic_s2_assembly(prompt, subtasks)
    assert text == text2
