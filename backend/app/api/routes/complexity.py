from fastapi import APIRouter, HTTPException, status
from app.schemas.complexity import ComplexityRequest, ComplexityResponse
from app.services.input_processor import InputProcessor
from app.services.complexity_analyzer import ComplexityAnalyzer

router = APIRouter()
input_processor = InputProcessor()
analyzer = ComplexityAnalyzer()

@router.post(
    "/complexity/analyze",
    response_model=ComplexityResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze prompt complexity factors and calculate weighted complexity score"
)
async def analyze_complexity_route(request: ComplexityRequest) -> ComplexityResponse:
    try:
        # Pre-process raw user prompt through InputProcessor
        processed = input_processor.process_prompt(request.text)
        return analyzer.analyze_complexity(processed.cleaned_prompt)
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
