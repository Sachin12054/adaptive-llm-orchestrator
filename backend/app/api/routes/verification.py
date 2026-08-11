from fastapi import APIRouter, HTTPException, status
from app.schemas.verification import (
    VerificationRequest,
    VerificationResponse,
    VerificationStatusResponse
)
from app.services.response_verifier import ResponseVerifier

router = APIRouter()
verifier = ResponseVerifier()

@router.get(
    "/verification/status",
    response_model=VerificationStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Response Verifier operational status and capability features"
)
async def get_verifier_status() -> VerificationStatusResponse:
    return verifier.get_status()

@router.post(
    "/verification/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate structural quality, completeness, and baseline prompt relevance of a response"
)
async def verify_response_route(request: VerificationRequest) -> VerificationResponse:
    try:
        return verifier.verify_response(request)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
