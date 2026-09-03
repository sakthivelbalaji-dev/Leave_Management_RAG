from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from leave_management.app.database.database import get_db
from leave_management.app.models import Employee, LeaveBalance, LeaveType
from leave_management.app.routers.dependencies import get_current_user, require_roles
from leave_management.app.schemas.leave_balance import (
    LeaveBalanceCreate, LeaveBalanceUpdate, LeaveBalanceResponse
)

router = APIRouter(prefix="/leave-balances", tags=["Leave Balances"])


def serialize(balance):
    return {
        "id": balance.id,
        "employee_id": balance.employee_id,
        "leave_type_id": balance.leave_type_id,
        "allocated_days": balance.allocated_days,
        "used_days": balance.used_days,
        "available_days": balance.available_days,
    }


@router.get("/me", response_model=list[LeaveBalanceResponse])
def my_balances(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    employee = db.query(Employee).filter(
        Employee.user_id == current_user.id
    ).first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee profile not found.")

    return [
        serialize(x)
        for x in db.query(LeaveBalance).filter(
            LeaveBalance.employee_id == employee.id
        ).all()
    ]


@router.get("/employee/{employee_id}", response_model=list[LeaveBalanceResponse])
def employee_balances(
    employee_id: int,
    current_user=Depends(require_roles("admin", "manager")),
    db: Session = Depends(get_db),
):
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found.")

    return [
        serialize(x)
        for x in db.query(LeaveBalance).filter(
            LeaveBalance.employee_id == employee_id
        ).all()
    ]


@router.post("", response_model=LeaveBalanceResponse, status_code=201)
def create_balance(
    payload: LeaveBalanceCreate,
    current_user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    if not db.query(Employee).filter(
        Employee.id == payload.employee_id
    ).first():
        raise HTTPException(status_code=404, detail="Employee not found.")

    if not db.query(LeaveType).filter(
        LeaveType.id == payload.leave_type_id
    ).first():
        raise HTTPException(status_code=404, detail="Leave type not found.")

    existing = db.query(LeaveBalance).filter(
        LeaveBalance.employee_id == payload.employee_id,
        LeaveBalance.leave_type_id == payload.leave_type_id,
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Balance already exists.")

    balance = LeaveBalance(**payload.model_dump())
    db.add(balance)
    db.commit()
    db.refresh(balance)
    return serialize(balance)


@router.put("/{balance_id}", response_model=LeaveBalanceResponse)
def update_balance(
    balance_id: int,
    payload: LeaveBalanceUpdate,
    current_user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    balance = db.query(LeaveBalance).filter(
        LeaveBalance.id == balance_id
    ).first()

    if not balance:
        raise HTTPException(status_code=404, detail="Balance not found.")

    if payload.allocated_days < balance.used_days:
        raise HTTPException(
            status_code=400,
            detail="Allocated days cannot be less than used days.",
        )

    balance.allocated_days = payload.allocated_days
    db.commit()
    db.refresh(balance)
    return serialize(balance)
