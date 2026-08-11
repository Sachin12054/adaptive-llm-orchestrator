from fastapi import APIRouter, HTTPException, status
from app.schemas.resource import ResourceSnapshotResponse
from app.services.resource_analyzer import ResourceAnalyzer

router = APIRouter()
analyzer = ResourceAnalyzer()

@router.get(
    "/resource/snapshot",
    response_model=ResourceSnapshotResponse,
    status_code=status.HTTP_200_OK,
    summary="Collect real-time system resource snapshot (CPU, RAM, GPU, CUDA, Disk, Process, Runtime)"
)
@router.get(
    "/resources/snapshot",
    response_model=ResourceSnapshotResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False
)
async def get_resource_snapshot_route() -> ResourceSnapshotResponse:
    try:
        return analyzer.get_resource_snapshot()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to gather system resource snapshot: {str(e)}"
        )
