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
    
    # Phase 6 & RUN 6 Propensity & Execution Fields for Off-Policy Evaluation & Attribution
    rl_selected_action: Optional[int] = Field(None, description="Discrete action index proposed by RL policy (-1 if unavailable)")
    rl_selected_model: Optional[str] = Field(None, description="Model identifier proposed by RL policy")
    executed_action: Optional[int] = Field(None, description="Discrete action index actually executed (0-6 for valid generation actions)")
    executed_model: Optional[str] = Field(None, description="Model identifier actually executed")
    executed_provider: Optional[str] = Field(None, description="Provider identifier actually executed")
    is_valid_rl_sample: bool = Field(True, description="True if record is a valid RL observation (action >= 0, state_dim == 12)")
    
    behavior_action: Optional[int] = Field(None, description="Discrete action index of behavior policy")
    behavior_model_id: Optional[str] = Field(None, description="Selected model identifier of behavior policy")
    propensity_probability: Optional[float] = Field(None, description="Behavior policy propensity P(a|s)")
    candidate_action_probabilities: Optional[Dict[str, float]] = Field(None, description="Full action probability distribution P(a|s) over candidates")
    propensity_available: bool = Field(False, description="True if online logged propensity score P(a|s) is valid for IPS")
    
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
    action_dim: int = Field(8, description="Unified action space dimensionality K=8")
    persistence_path: Optional[str] = Field(None, description="Path to persistent JSONL log file")
