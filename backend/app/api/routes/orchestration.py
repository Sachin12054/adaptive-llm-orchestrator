from fastapi import APIRouter, HTTPException, status
from app.schemas.orchestration import (
    OrchestrationRequest,
    OrchestrationResponse,
    OrchestrationStatusResponse
)
from app.services.orchestration_pipeline import OrchestrationPipeline

router = APIRouter()
pipeline = OrchestrationPipeline()

@router.get(
    "/orchestrate/status",
    response_model=OrchestrationStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check End-to-End Orchestration Pipeline operational readiness and configuration status"
)
async def get_orchestration_status() -> OrchestrationStatusResponse:
    return pipeline.get_status()

@router.post(
    "/orchestrate",
    response_model=OrchestrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute full End-to-End Adaptive Orchestration Pipeline (Decision -> Generation -> Verification -> Reward)"
)
async def run_orchestration_route(request: OrchestrationRequest) -> OrchestrationResponse:
    try:
        return pipeline.run_pipeline(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post(
    "/orchestrate/stream",
    summary="Stream End-to-End Orchestration Stage Execution Events via Server-Sent Events (SSE)"
)
async def run_orchestration_stream_route(request: OrchestrationRequest):
    from fastapi.responses import StreamingResponse
    try:
        return StreamingResponse(
            pipeline.run_pipeline_stream(request),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

