from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from leave_management.app.database.database import get_db
from leave_management.app.routers.dependencies import get_current_user
from leave_management.app.services.dashboard_service import (
    employee_dashboard, manager_dashboard, admin_dashboard
)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/me")
def dashboard_me(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role == "admin":
        return admin_dashboard(db)

    if current_user.role == "manager":
        return manager_dashboard(db, current_user.id)

    data = employee_dashboard(db, current_user.id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Employee profile not found.",
        )
    return data


@router.get("/manager")
def dashboard_manager(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role not in {"manager", "admin"}:
        raise HTTPException(status_code=403, detail="Manager access required.")
    return manager_dashboard(db, current_user.id)


@router.get("/admin")
def dashboard_admin(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return admin_dashboard(db)
