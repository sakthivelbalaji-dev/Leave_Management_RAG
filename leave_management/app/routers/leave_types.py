from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from leave_management.app.database.database import get_db
from leave_management.app.models import LeaveType
from leave_management.app.routers.dependencies import (
    get_current_user,
    require_roles,
)
from leave_management.app.schemas.leave_type import (
    LeaveTypeCreate,
    LeaveTypeUpdate,
    LeaveTypeResponse,
)


router = APIRouter(
    prefix="/leave-types",
    tags=["Leave Types"],
)


# ============================================================
# LIST ACTIVE LEAVE TYPES
# ============================================================

@router.get(
    "",
    response_model=list[LeaveTypeResponse],
)
def list_leave_types(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(LeaveType)
        .filter(LeaveType.is_active.is_(True))
        .order_by(LeaveType.name)
        .all()
    )


# ============================================================
# GET LEAVE TYPE
# ============================================================

@router.get(
    "/{leave_type_id}",
    response_model=LeaveTypeResponse,
)
def get_leave_type(
    leave_type_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = (
        db.query(LeaveType)
        .filter(LeaveType.id == leave_type_id)
        .first()
    )

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Leave type not found.",
        )

    return item


# ============================================================
# CREATE LEAVE TYPE
# ADMIN ONLY
# ============================================================

@router.post(
    "",
    response_model=LeaveTypeResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_leave_type(
    payload: LeaveTypeCreate,
    current_user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    name = payload.name.strip()

    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Leave type name cannot be empty.",
        )

    # Case-insensitive duplicate check
    existing = (
        db.query(LeaveType)
        .filter(LeaveType.name.ilike(name))
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Leave type already exists.",
        )

    item = LeaveType(
        name=name,
        description=payload.description,
        is_active=True,
    )

    db.add(item)

    try:
        db.commit()
        db.refresh(item)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create leave type.",
        )

    return item


# ============================================================
# UPDATE LEAVE TYPE
# ADMIN ONLY
# ============================================================

@router.put(
    "/{leave_type_id}",
    response_model=LeaveTypeResponse,
)
def update_leave_type(
    leave_type_id: int,
    payload: LeaveTypeUpdate,
    current_user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    item = (
        db.query(LeaveType)
        .filter(LeaveType.id == leave_type_id)
        .first()
    )

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Leave type not found.",
        )

    update_data = payload.model_dump(
        exclude_unset=True
    )

    # --------------------------------------------------------
    # NAME VALIDATION
    # --------------------------------------------------------

    if "name" in update_data:

        new_name = update_data["name"]

        if new_name is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Leave type name cannot be null.",
            )

        new_name = new_name.strip()

        if not new_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Leave type name cannot be empty.",
            )

        duplicate = (
            db.query(LeaveType)
            .filter(
                LeaveType.id != leave_type_id,
                LeaveType.name.ilike(new_name),
            )
            .first()
        )

        if duplicate:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Another leave type with this name already exists.",
            )

        update_data["name"] = new_name

    # --------------------------------------------------------
    # APPLY UPDATE
    # --------------------------------------------------------

    for key, value in update_data.items():
        setattr(item, key, value)

    try:
        db.commit()
        db.refresh(item)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update leave type.",
        )

    return item


# ============================================================
# DEACTIVATE LEAVE TYPE
# ADMIN ONLY
# ============================================================

@router.delete(
    "/{leave_type_id}",
)
def delete_leave_type(
    leave_type_id: int,
    current_user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    item = (
        db.query(LeaveType)
        .filter(LeaveType.id == leave_type_id)
        .first()
    )

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Leave type not found.",
        )

    if not item.is_active:
        return {
            "success": True,
            "message": "Leave type is already inactive.",
        }

    item.is_active = False

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate leave type.",
        )

    return {
        "success": True,
        "message": "Leave type deactivated successfully.",
    }