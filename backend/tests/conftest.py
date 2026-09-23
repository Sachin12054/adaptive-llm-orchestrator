import os
import tempfile
from pathlib import Path

import pytest


_TEST_BUFFER_DIR = Path(tempfile.mkdtemp(prefix="adaptive-llm-tests-"))
_TEST_BUFFER_PATH = _TEST_BUFFER_DIR / "experience_buffer.jsonl"
os.environ["ADAPTIVE_TEST_BUFFER_PATH"] = str(_TEST_BUFFER_PATH)


@pytest.fixture(autouse=True)
def isolate_default_experience_buffer():
    from app.services.experience_buffer import ExperienceBufferService

    ExperienceBufferService._instance = None
    yield
    instance = ExperienceBufferService._instance
    if instance is not None:
        instance.clear()
    ExperienceBufferService._instance = None