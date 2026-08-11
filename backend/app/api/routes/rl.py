import os
import sys
from fastapi import APIRouter, HTTPException, status
from typing import List

from app.schemas.rl import (
    RLStatusResponse,
    RLTrainRequest,
    RLTrainResponse,
    RLEvalRequest,
    RLEvalResponse
)
from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.services.experience_buffer import ExperienceBufferService, ACTION_MAP

# Add project root to sys.path to access rl.policy_trainer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rl.policy_trainer import PolicyTrainer

router = APIRouter()

@router.get(
    "/rl/status",
    response_model=RLStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check RL Policy status and loaded weights metadata"
)
async def get_rl_status() -> RLStatusResponse:
    policy = RLContextualBanditPolicy()
    buffer_service = ExperienceBufferService()
    
    loaded = (policy.weights is not None)
    current_status = "ready" if loaded else "policy_not_available"

    return RLStatusResponse(
        status=current_status,
        service="rl_policy_framework",
        policy_loaded=loaded,
        policy_name=policy.policy_name,
        state_dim=12,
        action_map=ACTION_MAP,
        policy_version=policy.policy_version,
        training_samples_count=policy.training_samples_count,
        mean_squared_error=policy.mean_squared_error,
        last_trained_timestamp=policy.last_trained_timestamp
    )

@router.post(
    "/rl/train",
    response_model=RLTrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger offline batch training of the Linear Contextual Bandit policy on stored Step 20 experiences"
)
async def train_rl_policy(request: RLTrainRequest) -> RLTrainResponse:
    trainer = PolicyTrainer()
    res = trainer.train_policy(request)
    return res

@router.post(
    "/rl/eval",
    response_model=RLEvalResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate shadow RL policy agreement ratio and prediction metrics against baseline decisions"
)
async def evaluate_rl_policy(request: RLEvalRequest) -> RLEvalResponse:
    policy = RLContextualBanditPolicy()
    buffer_service = ExperienceBufferService()

    if policy.weights is None:
        return RLEvalResponse(
            success=False,
            status="policy_not_available",
            policy_loaded=False,
            total_evaluated_samples=0,
            agreement_count=0,
            agreement_ratio=0.0,
            off_policy_reward_estimate="unavailable",
            mean_baseline_reward=0.0,
            mean_rl_predicted_reward=0.0
        )

    experiences = buffer_service._buffer
    if not experiences:
        return RLEvalResponse(
            success=False,
            status="no_experiences",
            policy_loaded=True,
            total_evaluated_samples=0,
            agreement_count=0,
            agreement_ratio=0.0,
            off_policy_reward_estimate="unavailable",
            mean_baseline_reward=0.0,
            mean_rl_predicted_reward=0.0
        )

    eval_samples = experiences[-request.eval_batch_size:]
    agreement_count = 0
    baseline_rewards = []
    rl_predicted_rewards = []

    for rec in eval_samples:
        baseline_rewards.append(rec.reward)
        
        # Action Map reverse lookup
        rec_baseline_model = rec.action_model_id
        
        # Evaluate shadow prediction on rec.state (mock candidate list)
        s_vec = rec.state
        if len(s_vec) == 12:
            raw_scores = float(policy.weights[rec.action] @ s_vec + policy.bias[rec.action]) if rec.action >= 0 and rec.action < policy.weights.shape[0] else 0.0
            rl_predicted_rewards.append(raw_scores)
            
            # Predict top action
            scores = policy.weights @ s_vec + policy.bias
            top_a = int(scores.argmax())
            top_model = policy.reverse_action_map.get(top_a)
            if top_model == rec_baseline_model:
                agreement_count += 1

    total_eval = len(eval_samples)
    agreement_ratio = round(agreement_count / total_eval, 4) if total_eval > 0 else 0.0
    mean_base_r = round(float(sum(baseline_rewards) / total_eval), 4) if total_eval > 0 else 0.0
    mean_rl_pred = round(float(sum(rl_predicted_rewards) / len(rl_predicted_rewards)), 4) if rl_predicted_rewards else 0.0

    return RLEvalResponse(
        success=True,
        status="completed",
        policy_loaded=True,
        total_evaluated_samples=total_eval,
        agreement_count=agreement_count,
        agreement_ratio=agreement_ratio,
        off_policy_reward_estimate="unavailable",
        mean_baseline_reward=mean_base_r,
        mean_rl_predicted_reward=mean_rl_pred
    )
