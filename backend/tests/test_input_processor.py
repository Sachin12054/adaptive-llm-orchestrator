import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.input_processor import InputProcessor

client = TestClient(app)

def test_normal_prompt():
    processor = InputProcessor()
    raw_prompt = "What is the capital of France?"
    result = processor.process_prompt(raw_prompt)
    assert result.cleaned_prompt == "What is the capital of France?"
    assert result.original_prompt == raw_prompt
    assert result.character_count == 30
    assert result.word_count == 6

def test_leading_trailing_whitespace():
    processor = InputProcessor()
    raw_prompt = "  What is the capital of France?  "
    result = processor.process_prompt(raw_prompt)
    assert result.cleaned_prompt == "What is the capital of France?"
    assert result.original_prompt == raw_prompt

def test_multiple_internal_spaces():
    processor = InputProcessor()
    raw_prompt = "What   is  the   capital  of France?"
    result = processor.process_prompt(raw_prompt)
    assert result.cleaned_prompt == "What is the capital of France?"
    assert result.original_prompt == raw_prompt

def test_empty_prompt():
    processor = InputProcessor()
    with pytest.raises(ValueError, match="cannot be empty"):
        processor.process_prompt("")

def test_whitespace_only_prompt():
    processor = InputProcessor()
    with pytest.raises(ValueError, match="cannot be empty"):
        processor.process_prompt("   \t\n ")

def test_max_prompt_length_boundary():
    processor = InputProcessor(max_prompt_length=10)
    raw_prompt = "1234567890"  # Exactly 10 chars
    result = processor.process_prompt(raw_prompt)
    assert result.cleaned_prompt == "1234567890"

def test_prompt_exceeding_max_length():
    processor = InputProcessor(max_prompt_length=10)
    raw_prompt = "12345678901"  # 11 chars
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        processor.process_prompt(raw_prompt)

def test_original_prompt_preserved():
    processor = InputProcessor()
    raw_prompt = "   Hello   World! \n "
    result = processor.process_prompt(raw_prompt)
    assert result.original_prompt == raw_prompt
    assert result.cleaned_prompt == "Hello World!"

def test_word_count_accuracy():
    processor = InputProcessor()
    raw_prompt = "One two three four five"
    result = processor.process_prompt(raw_prompt)
    assert result.word_count == 5

def test_character_count_accuracy():
    processor = InputProcessor()
    raw_prompt = "Hello World"
    result = processor.process_prompt(raw_prompt)
    assert result.character_count == 11

def test_api_input_process_valid():
    payload = {"prompt": "  What is the capital of France?  "}
    response = client.post("/api/input/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["original_prompt"] == "  What is the capital of France?  "
    assert data["cleaned_prompt"] == "What is the capital of France?"
    assert data["character_count"] == 30
    assert data["word_count"] == 6

def test_api_input_process_empty_invalid():
    payload = {"prompt": "   "}
    response = client.post("/api/input/process", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "cannot be empty" in data["detail"]
