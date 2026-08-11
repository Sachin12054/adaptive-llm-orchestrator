from fastapi import APIRouter, HTTPException, status
from app.schemas.embedding import EmbeddingRequest, EmbeddingResponse
from app.services.embedding_service import EmbeddingService

router = APIRouter()

@router.post(
    "/embedding/generate",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate semantic embedding metadata for input text"
)
async def generate_embedding_route(request: EmbeddingRequest) -> EmbeddingResponse:
    try:
        service = EmbeddingService()
        return service.generate_embedding(
            text=request.text,
            include_vector=request.include_vector
        )
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
