from typing import List, Dict
from pydantic import BaseModel, Field

class IntentRequest(BaseModel):
    text: str = Field(..., description="Prompt text to analyze for semantic intent")

class RankedIntent(BaseModel):
    intent: str = Field(..., description="Intent label")
    score: float = Field(..., description="Aggregated cosine similarity score for this intent")

class IntentResponse(BaseModel):
    text: str = Field(..., description="Input text analyzed")
    intent: str = Field(..., description="Top predicted semantic intent")
    top_similarity: float = Field(..., description="Highest aggregated prototype cosine similarity score")
    second_similarity: float = Field(..., description="Second-highest aggregated prototype cosine similarity score")
    margin: float = Field(..., description="Separation margin (top_similarity - second_similarity)")
    is_ambiguous: bool = Field(..., description="True if separation margin is below ambiguity threshold")
    ranked_intents: List[RankedIntent] = Field(..., description="Ranked list of all candidate intents and similarity scores")
    embedding_model: str = Field(..., description="Embedding model identifier used")
    embedding_dimension: int = Field(..., description="Dimension of embedding vectors used for similarity")
    latency_ms: float = Field(..., description="Total intent classification pipeline latency in milliseconds")
    breakdown_latency_ms: Dict[str, float] = Field(..., description="Detailed latency breakdown (embedding, classification)")
