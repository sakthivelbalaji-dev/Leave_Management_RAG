from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from sqlalchemy.orm import Session

from leave_management.app.database.database import (
    get_db,
)

from leave_management.app.routers.dependencies import (
    get_current_user,
)

from leave_management.app.schemas.ai import (
    AIQueryRequest,
    AIQueryResponse,
    ModelEvaluationRequest,
)

from rag.initializer import rag_pipeline


router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)


# ============================================================
# NORMAL AI QUERY
# ============================================================

@router.post(
    "/query",
    response_model=AIQueryResponse,
)
def query_ai(
    request: AIQueryRequest,
    current_user=Depends(
        get_current_user
    ),
    db: Session = Depends(
        get_db
    ),
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
            conversation_history=request.conversation_history,
        )

    except HTTPException:
        raise

    except Exception as exc:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "AI service error: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# EMBEDDING MODEL EVALUATION
# ============================================================

@router.post(
    "/evaluate",
)
def evaluate_models(
    request: ModelEvaluationRequest,
    current_user=Depends(
        get_current_user
    ),
):

    try:

        # ----------------------------------------------------
        # GET COMPARISON RETRIEVER
        # ----------------------------------------------------

        comparison_retriever = (
            rag_pipeline.comparison_retriever
        )

        if comparison_retriever is None:

            raise HTTPException(
                status_code=503,
                detail=(
                    "Embedding comparison "
                    "retriever is not initialized."
                ),
            )

        result = comparison_retriever.compare(
            query=request.question,
            top_k=request.top_k,
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Model evaluation error: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
# ============================================================
# LIVE EMBEDDING COMPARISON
# ============================================================

@router.post(
    "/compare",
)
def compare_models(
    request: ModelEvaluationRequest,
    current_user=Depends(
        get_current_user
    ),
):

    try:

        comparison_retriever = (
            rag_pipeline.comparison_retriever
        )

        if comparison_retriever is None:

            raise HTTPException(
                status_code=503,
                detail=(
                    "Embedding comparison "
                    "retriever is not initialized."
                ),
            )

        result = (
            comparison_retriever.compare(
                query=request.question,
                top_k=request.top_k,
            )
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Embedding comparison error: "
                f"{type(exc).__name__}: {exc}"
            ),
        )