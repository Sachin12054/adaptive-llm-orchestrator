from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str = Field(..., example="adaptive-llm-orchestrator")
