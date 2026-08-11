from typing import List, Optional
from pydantic import BaseModel, Field

class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Cleaned prompt text to embed")
    include_vector: bool = Field(
        False,
        description="Optional debug flag to include the full raw embedding vector in response"
    )

class EmbeddingResponse(BaseModel):
    model: str = Field(..., description="Embedding model identifier used")
    dimension: int = Field(..., description="Dimension of generated embedding vector")
    latency_ms: float = Field(..., description="Measured embedding generation time in milliseconds")
    device: str = Field(..., description="Target execution device (cpu or cuda)")
    embedding: Optional[List[float]] = Field(
        None,
        description="Optional full 1024-dim embedding vector (only populated if include_vector=True)"
    )
