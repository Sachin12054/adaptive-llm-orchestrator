from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ShadowDecision(BaseModel):
    policy_name: str = Field("rl_contextual_bandit_policy", description="Shadow policy name")
    proposed_model: Optional[str] = Field(None, description="Model proposed by shadow RL policy")
    predicted_reward: float = Field(0.0, description="Predicted expected reward Q_hat(s, a)")
    action_index: int = Field(-1, description="Discrete action index proposed by RL policy")
    agrees_with_baseline: bool = Field(False, description="True if shadow RL proposal matches baseline decision")
    action_masked: bool = Field(False, description="True if shadow policy encountered an action mask")
    policy_available: bool = Field(False, description="True if trained RL policy weights are loaded")

class RLStatusResponse(BaseModel):
    status: str = Field(..., description="RL service status: ready, policy_not_available, insufficient_data")
    service: str = Field("rl_policy_framework", description="Service name")
    policy_loaded: bool = Field(..., description="True if trained policy weights are currently loaded")
    policy_name: str = Field("rl_contextual_bandit_policy", description="RL policy algorithm name")
    state_dim: int = Field(12, description="Expected state dimension")
    action_map: Dict[str, int] = Field(..., description="Discrete action index mapping")
    policy_version: Optional[str] = Field(None, description="Trained policy version string")
    training_samples_count: int = Field(0, description="Number of samples used in last training run")
    mean_squared_error: Optional[float] = Field(None, description="MSE loss of loaded policy")
    last_trained_timestamp: Optional[float] = Field(None, description="Timestamp of last policy training run")

class RLTrainRequest(BaseModel):
    minimum_samples: int = Field(100, ge=10, description="Minimum number of Step 20 experiences required for training")
    learning_rate: float = Field(0.01, gt=0.0, description="Gradient descent learning rate for linear weights")
    epochs: int = Field(50, ge=1, description="Number of training epochs over stored experiences")
    l2_lambda: float = Field(0.01, ge=0.0, description="L2 regularization parameter")

class RLTrainResponse(BaseModel):
    success: bool = Field(..., description="True if training completed successfully")
    status: str = Field(..., description="Training status: completed, insufficient_data, failed")
    message: str = Field(..., description="Human-readable result summary")
    samples_used: int = Field(0, description="Number of Step 20 experience records trained on")
    minimum_required: int = Field(100, description="Configured minimum sample threshold")
    final_mse: Optional[float] = Field(None, description="Final Mean Squared Error loss")
    policy_path: Optional[str] = Field(None, description="Path where trained policy was saved")

class RLEvalRequest(BaseModel):
    eval_batch_size: int = Field(50, ge=1, description="Number of stored experiences to evaluate shadow policy on")

class RLEvalResponse(BaseModel):
    success: bool = Field(..., description="True if evaluation completed successfully")
    status: str = Field(..., description="Evaluation status")
    policy_loaded: bool = Field(..., description="True if RL policy weights are active")
    total_evaluated_samples: int = Field(0, description="Number of samples evaluated")
    agreement_count: int = Field(0, description="Number of decisions where RL shadow proposal matched baseline")
    agreement_ratio: float = Field(0.0, description="Percentage of matching decisions [0.0 - 1.0]")
    off_policy_reward_estimate: str = Field("unavailable", description="Off-policy reward estimate (unavailable due to deterministic baseline propensity)")
    mean_baseline_reward: float = Field(0.0, description="Mean observed baseline reward across evaluated samples")
    mean_rl_predicted_reward: float = Field(0.0, description="Mean predicted Q_hat reward across evaluated samples")
