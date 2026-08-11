from urllib.parse import unquote
from fastapi import APIRouter, HTTPException, status
from app.schemas.model import ModelMetadata, ModelRegistryResponse
from app.services.model_registry import ModelRegistry

router = APIRouter()
registry = ModelRegistry()

@router.get(
    "/models",
    response_model=ModelRegistryResponse,
    status_code=status.HTTP_200_OK,
    summary="List all registered model metadata and availability status"
)
async def list_registered_models() -> ModelRegistryResponse:
    models = registry.list_models()
    return ModelRegistryResponse(
        models=models,
        total_count=len(models)
    )

@router.get(
    "/models/{model_id:path}",
    response_model=ModelMetadata,
    status_code=status.HTTP_200_OK,
    summary="Retrieve metadata for a specific registered model by ID"
)
async def get_model_metadata(model_id: str) -> ModelMetadata:
    decoded_id = unquote(model_id)
    model = registry.get_model(decoded_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{decoded_id}' not found in registry."
        )
    return model
