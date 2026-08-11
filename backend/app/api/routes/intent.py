from fastapi import APIRouter, HTTPException, status
from app.schemas.intent import IntentRequest, IntentResponse
from app.services.input_processor import InputProcessor
from app.services.intent_classifier import IntentClassifier

router = APIRouter()
input_processor = InputProcessor()
classifier = IntentClassifier()

@router.post(
    "/intent/classify",
    response_model=IntentResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantically classify user prompt intent via BGE-M3 prototype similarity"
)
async def classify_intent_route(request: IntentRequest) -> IntentResponse:
    try:
        # Pre-process raw user prompt through InputProcessor
        processed = input_processor.process_prompt(request.text)
        return classifier.classify_intent(processed.cleaned_prompt)
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
