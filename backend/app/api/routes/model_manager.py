from fastapi import APIRouter, HTTPException, status
from app.schemas.model_manager import (
    ModelExecutionRequest,
    ModelExecutionResponse,
    ModelManagerStatusResponse
)
from app.services.model_manager import ModelManager

router = APIRouter()
manager = ModelManager()

@router.get(
    "/model-manager/status",
    response_model=ModelManagerStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve ModelManager operational readiness and executable models status"
)
async def get_model_manager_status() -> ModelManagerStatusResponse:
    return manager.get_status()

@router.post(
    "/model-manager/execute",
    response_model=ModelExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute specific requested LLM model through provider adapter"
)
async def execute_model_route(request: ModelExecutionRequest) -> ModelExecutionResponse:
    response = manager.execute(request)
    if not response.success and response.execution_status == "unsupported_model":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.error_message
        )
    return response
