from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ModelMetadata(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    provider: str = Field(..., description="Model vendor or host provider (e.g. Google Gemini, BAAI)")
    display_name: str = Field(..., description="Human-readable model name")
    model_type: str = Field(..., description="Model classification type: embedding or llm")
    capabilities: List[str] = Field(..., description="List of supported functional capabilities")
    context_length: Optional[int] = Field(None, description="Maximum context window size in tokens if known")
    embedding_dimension: Optional[int] = Field(None, description="Vector dimension size for embedding models")
    execution_mode: str = Field(..., description="Execution mode: local or online_api")
    local: bool = Field(..., description="True if model executes on local hardware")
    available: bool = Field(..., description="True if model is installed or API key is configured")
    configuration_status: str = Field(..., description="Status: configured, not_configured, or partially_configured")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Hardware/software prerequisites or environment variables")
    input_cost_per_1k: float = Field(0.0, description="Cost per 1,000 input tokens in USD")
    output_cost_per_1k: float = Field(0.0, description="Cost per 1,000 output tokens in USD")
    metadata_source: str = Field(..., description="Origin source of metadata specification")

class ModelRegistryResponse(BaseModel):
    models: List[ModelMetadata] = Field(..., description="List of registered model metadata entries")
    total_count: int = Field(..., description="Total number of registered models in registry")
