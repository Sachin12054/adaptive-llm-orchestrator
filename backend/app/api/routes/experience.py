from fastapi import APIRouter, HTTPException, Query, status
from app.schemas.experience import (
    ExperienceRecord,
    ExperienceBatchResponse,
    ExperienceBufferStatusResponse
)
from app.services.experience_buffer import ExperienceBufferService

router = APIRouter()
buffer_service = ExperienceBufferService()

@router.get(
    "/experience/status",
    response_model=ExperienceBufferStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Experience Replay Buffer status and current memory size"
)
async def get_experience_buffer_status() -> ExperienceBufferStatusResponse:
    return buffer_service.get_status()

@router.get(
    "/experience/sample",
    response_model=ExperienceBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Sample a random minibatch of transition experience records from the replay buffer"
)
async def sample_experience_batch(batch_size: int = Query(32, ge=1, description="Minibatch size to sample")) -> ExperienceBatchResponse:
    try:
        samples = buffer_service.sample_batch(batch_size)
        return ExperienceBatchResponse(
            experiences=samples,
            batch_size=len(samples)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
