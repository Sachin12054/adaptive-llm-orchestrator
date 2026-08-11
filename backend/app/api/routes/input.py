from fastapi import APIRouter, HTTPException, status
from app.schemas.input import ProcessPromptRequest, ProcessedPromptResponse
from app.services.input_processor import InputProcessor

router = APIRouter()
processor = InputProcessor()

@router.post(
    "/input/process",
    response_model=ProcessedPromptResponse,
    status_code=status.HTTP_200_OK,
    summary="Process and sanitize raw user prompt"
)
async def process_input_prompt(request: ProcessPromptRequest) -> ProcessedPromptResponse:
    try:
        return processor.process_prompt(request.prompt)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
