from fastapi import APIRouter, status
from typing import Dict, Any
from app.services.experience_buffer import ExperienceBufferService

router = APIRouter()
buffer_service = ExperienceBufferService()

@router.get(
    "/metrics/cost",
    status_code=status.HTTP_200_OK,
    summary="Get authoritative backend API cost telemetry and model/provider breakdown"
)
@router.get(
    "/v1/metrics/cost",
    status_code=status.HTTP_200_OK,
    summary="Get authoritative backend API cost telemetry and model/provider breakdown"
)
async def get_cost_metrics() -> Dict[str, Any]:
    return buffer_service.get_cost_metrics()
