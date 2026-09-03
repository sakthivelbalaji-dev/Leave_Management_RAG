from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from leave_management.app.database.database import get_db
from leave_management.app.routers.dependencies import get_current_user
from leave_management.app.schemas.ai import AIQueryRequest, AIQueryResponse

from rag.initializer import rag_pipeline


router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)


@router.post(
    "/query",
    response_model=AIQueryResponse,
)
def query_ai(
    request: AIQueryRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        return rag_pipeline.query(
            question=request.question,
            db=db,
            user_id=current_user.id,
            user_role=current_user.role,
            top_k=request.top_k,
            draft=request.draft,
            confirmed=request.confirmed,
        )

    except HTTPException:
        raise

    except Exception as exc:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"AI service error: {type(exc).__name__}: {exc}",
        )
