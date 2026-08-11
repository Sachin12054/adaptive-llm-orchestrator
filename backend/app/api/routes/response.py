from fastapi import APIRouter, HTTPException, status
from app.schemas.response import (
    ResponseGenerationRequest,
    ResponseGenerationResponse,
    ResponseStatusResponse
)
from app.services.response_generator import ResponseGenerator

router = APIRouter()
generator = ResponseGenerator()

@router.get(
    "/response/status",
    response_model=ResponseStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Response Generator status and underlying ModelManager availability"
)
async def get_response_generator_status() -> ResponseStatusResponse:
    return generator.get_status()

@router.post(
    "/response/generate",
    response_model=ResponseGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Coordinate response generation for selected model via ModelManager"
)
async def generate_response_route(request: ResponseGenerationRequest) -> ResponseGenerationResponse:
    try:
        response = generator.generate_response(request)
        if not response.success and response.execution_status == "unsupported_model":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response.error_message
            )
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
