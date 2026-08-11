from fastapi import APIRouter, HTTPException, status
from app.schemas.reward import (
    RewardComputeRequest,
    RewardComputeResponse,
    RewardStatusResponse
)
from app.services.reward_signal import RewardSignal

router = APIRouter()
reward_service = RewardSignal()

@router.get(
    "/reward/status",
    response_model=RewardStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Reward Signal service operational status and learning framework readiness"
)
async def get_reward_status() -> RewardStatusResponse:
    return reward_service.get_status()

@router.post(
    "/reward/compute",
    response_model=RewardComputeResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute deterministic baseline reward signal and component breakdown from pipeline outputs"
)
async def compute_reward_route(request: RewardComputeRequest) -> RewardComputeResponse:
    try:
        return reward_service.compute_reward(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
