from fastapi import APIRouter, HTTPException, status
from app.schemas.decision import (
    DecisionRequest,
    DecisionResponse,
    DecisionStatusResponse
)
from app.services.adaptive_decision_engine import AdaptiveDecisionEngine

router = APIRouter()
engine = AdaptiveDecisionEngine()

@router.get(
    "/decision/status",
    response_model=DecisionStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Adaptive Decision Engine operational readiness and active policy status"
)
async def get_decision_engine_status() -> DecisionStatusResponse:
    return engine.get_status()

@router.post(
    "/decision/decide",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate input prompt and determine optimal model routing selection"
)
async def decide_model_route(request: DecisionRequest) -> DecisionResponse:
    try:
        return engine.decide(request)
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
