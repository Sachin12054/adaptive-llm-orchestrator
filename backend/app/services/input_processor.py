import re
from app.core.config import settings
from app.schemas.input import ProcessedPromptResponse

class InputProcessor:
    def __init__(self, max_prompt_length: int = None):
        self.max_prompt_length = (
            max_prompt_length if max_prompt_length is not None else settings.MAX_PROMPT_LENGTH
        )

    def process_prompt(self, prompt: str) -> ProcessedPromptResponse:
        if prompt is None or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-null string.")

        # Check for empty or whitespace-only prompt
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty or contain only whitespace.")

        # Check for maximum length violation without silent truncation
        if len(prompt) > self.max_prompt_length:
            raise ValueError(
                f"Prompt length ({len(prompt)}) exceeds maximum allowed limit of {self.max_prompt_length} characters."
            )

        original_prompt = prompt
        # Normalize whitespace (strip leading/trailing and consolidate repeated spaces/newlines)
        cleaned_prompt = re.sub(r"\s+", " ", prompt.strip())
        character_count = len(cleaned_prompt)
        word_count = len(cleaned_prompt.split()) if cleaned_prompt else 0

        return ProcessedPromptResponse(
            original_prompt=original_prompt,
            cleaned_prompt=cleaned_prompt,
            character_count=character_count,
            word_count=word_count,
        )
