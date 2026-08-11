from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ExperienceRecord(BaseModel):
    experience_id: str = Field(..., description="Unique experience record identifier (e.g. exp-9f8a7b6c)")
    state: List[float] = Field(..., description="Normalized numerical state vector representation (fixed dimension D)")
    action: int = Field(..., description="Discrete action index corresponding to selected LLM model")
    action_model_id: str = Field(..., description="Selected LLM model identifier string")
    reward: float = Field(..., description="Exact Step 18 scalar reward signal [0.0 - 1.0]")
    next_state: Optional[List[float]] = Field(None, description="Next state vector (explicitly None for single-turn requests)")
    done: bool = Field(True, description="True for terminal single-turn requests")
    timestamp: float = Field(..., description="POSIX creation timestamp in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Auditability metadata (intent, complexity, telemetry, status, latency)")

class ExperienceRecordRequest(BaseModel):
    prompt: str = Field(..., description="User prompt text")
    selected_model: Optional[str] = Field(None, description="Selected model identifier")
    reward: float = Field(..., description="Step 18 scalar reward")

class ExperienceBatchResponse(BaseModel):
    experiences: List[ExperienceRecord] = Field(..., description="Sampled batch of experience records")
    batch_size: int = Field(..., description="Number of experiences in sampled batch")

class ExperienceBufferStatusResponse(BaseModel):
    status: str = Field("ready", description="Buffer operational status")
    service: str = Field("experience_buffer", description="Service identifier")
    capacity: int = Field(..., description="Maximum buffer capacity limit")
    current_size: int = Field(..., description="Current number of stored experiences")
    state_dim: int = Field(..., description="Fixed dimensionality D of encoded state vectors")
    persistence_path: Optional[str] = Field(None, description="Path to persistent JSONL log file")
