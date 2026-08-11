from typing import Dict
from pydantic import BaseModel, Field

class ComplexityRequest(BaseModel):
    text: str = Field(..., description="Prompt text to analyze for request complexity")

class ComplexityFactors(BaseModel):
    semantic_complexity: float = Field(..., description="Normalized BGE-M3 prototype similarity complexity score [0.0 - 1.0]")
    reasoning_complexity: float = Field(..., description="Normalized conceptual reasoning depth score [0.0 - 1.0]")
    task_complexity: float = Field(..., description="Normalized task/operation multiplicity score [0.0 - 1.0]")
    context_complexity: float = Field(..., description="Normalized input volume & structural density score [0.0 - 1.0]")
    output_complexity: float = Field(..., description="Normalized expected response scope/formatting score [0.0 - 1.0]")

class ComplexityResponse(BaseModel):
    text: str = Field(..., description="Input text analyzed")
    complexity_level: str = Field(..., description="Calculated complexity level: low, medium, high, very_high")
    complexity_score: float = Field(..., description="Transparent weighted overall complexity score [0.0 - 1.0]")
    factors: ComplexityFactors = Field(..., description="Individual normalized factor score breakdown")
    task_count: int = Field(..., description="Estimated number of distinct requested operations/tasks")
    estimated_reasoning_depth: int = Field(..., description="Estimated conceptual reasoning level (0 to 4)")
    embedding_model: str = Field(..., description="Embedding model identifier used")
    embedding_dimension: int = Field(..., description="Dimension of embedding vectors used for analysis")
    latency_ms: float = Field(..., description="Total complexity analysis pipeline latency in milliseconds")
    breakdown_latency_ms: Dict[str, float] = Field(..., description="Detailed latency breakdown (embedding, analysis)")
