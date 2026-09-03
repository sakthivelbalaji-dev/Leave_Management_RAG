from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from leave_management.app.database.database import get_db
from leave_management.app.models import Employee, User
from leave_management.app.routers.dependencies import get_current_user, require_roles
from leave_management.app.schemas.user import UserAdminResponse

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me/profile")
def my_profile(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    employee = db.query(Employee).filter(
        Employee.user_id == current_user.id
    ).first()

    return {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "role": current_user.role,
            "is_active": current_user.is_active,
        },
        "employee": None if not employee else {
            "id": employee.id,
            "employee_code": employee.employee_code,
            "full_name": employee.full_name,
            "department": employee.department,
            "manager_id": employee.manager_id,
        },
    }


@router.get("", response_model=list[UserAdminResponse])
def list_users(
    current_user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    rows = db.query(User).all()
    result = []

    for u in rows:
        e = db.query(Employee).filter(Employee.user_id == u.id).first()
        result.append(UserAdminResponse(
            id=u.id,
            username=u.username,
            email=u.email,
            role=u.role,
            is_active=u.is_active,
            employee_id=e.id if e else None,
            employee_code=e.employee_code if e else None,
            full_name=e.full_name if e else None,
            department=e.department if e else None,
            manager_id=e.manager_id if e else None,
        ))

    return result


@router.patch("/{user_id}/role")
def change_role(
    user_id: int,
    role: str,
    current_user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    if role not in {"employee", "manager", "admin"}:
        raise HTTPException(status_code=400, detail="Invalid role.")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.role = role
    db.commit()
    return {"success": True, "message": "Role updated.", "role": role}


@router.patch("/{user_id}/status")
def change_status(
    user_id: int,
    is_active: bool,
    current_user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.is_active = is_active
    db.commit()
    return {"success": True, "is_active": is_active}
