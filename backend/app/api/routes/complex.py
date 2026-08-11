from fastapi import APIRouter, HTTPException, status
from app.schemas.complex import ComplexPlanRequest, ComplexTaskPlan
from app.services.complex.complex_task_decomposer import ComplexTaskDecomposer

router = APIRouter()
decomposer = ComplexTaskDecomposer()

@router.post(
    "/complex/plan",
    response_model=ComplexTaskPlan,
    status_code=status.HTTP_200_OK,
    summary="Decompose complex user prompt into subtasks with model allocation and DAG execution levels"
)
async def generate_complex_plan(request: ComplexPlanRequest) -> ComplexTaskPlan:
    try:
        exec_mode = request.execution_mode or "local"
        return decomposer.decompose(
            request.prompt,
            execution_mode=exec_mode,
            primary_selected_model=request.primary_selected_model
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Complex task decomposition failed: {str(e)}"
        )
