from pydantic import BaseModel, Field

class ProcessPromptRequest(BaseModel):
    prompt: str = Field(..., description="Raw user prompt string to process")

class ProcessedPromptResponse(BaseModel):
    original_prompt: str = Field(..., description="Original, un-truncated raw prompt")
    cleaned_prompt: str = Field(..., description="Normalized prompt with stripped whitespace")
    character_count: int = Field(..., description="Character count of cleaned prompt")
    word_count: int = Field(..., description="Word count of cleaned prompt")
