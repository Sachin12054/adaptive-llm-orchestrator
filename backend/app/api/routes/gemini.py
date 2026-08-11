from fastapi import APIRouter, HTTPException, status
from app.schemas.provider import (
    ProviderGenerationRequest,
    ProviderGenerationResponse,
    ProviderStatusResponse
)
from app.services.providers.gemini_provider import GeminiProvider

router = APIRouter()
provider = GeminiProvider()

@router.get(
    "/providers/gemini/status",
    response_model=ProviderStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Google Gemini provider configuration and operational status"
)
async def get_gemini_status() -> ProviderStatusResponse:
    return provider.get_status()

@router.post(
    "/providers/gemini/generate",
    response_model=ProviderGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute text generation via specified Gemini model"
)
async def generate_gemini_content(request: ProviderGenerationRequest) -> ProviderGenerationResponse:
    response = provider.generate(request)
    if not response.success and "not registered" in (response.error_message or "").lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.error_message
        )
    return response
