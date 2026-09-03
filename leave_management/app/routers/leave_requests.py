from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from leave_management.app.database.database import get_db
from leave_management.app.models import Employee, LeaveRequest
from leave_management.app.routers.dependencies import get_current_user, require_roles
from leave_management.app.schemas.leave_request import (
    LeaveRequestCreate, LeaveDecision, LeaveRequestResponse
)
from leave_management.app.services.leave_service import (
    create_leave_request, approve_request, reject_request
)

router = APIRouter(prefix="/leave-requests", tags=["Leave Requests"])


def check_manager_access(db, current_user, request_id):
    item = db.query(LeaveRequest).filter(
        LeaveRequest.id == request_id
    ).first()

    if not item:
        raise HTTPException(status_code=404, detail="Leave request not found.")

    employee = db.query(Employee).filter(
        Employee.id == item.employee_id
    ).first()

    if current_user.role == "manager":
        if not employee or employee.manager_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="This request is not assigned to you.",
            )


@router.post("", response_model=LeaveRequestResponse, status_code=201)
def submit_leave_request(
    payload: LeaveRequestCreate,
    current_user=Depends(require_roles("employee", "manager", "admin")),
    db: Session = Depends(get_db),
):
    request, error = create_leave_request(db, current_user.id, payload)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return request


@router.get("/me", response_model=list[LeaveRequestResponse])
def my_requests(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    employee = db.query(Employee).filter(
        Employee.user_id == current_user.id
    ).first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee profile not found.")

    return db.query(LeaveRequest).filter(
        LeaveRequest.employee_id == employee.id
    ).order_by(LeaveRequest.created_at.desc()).all()


@router.get("/pending", response_model=list[LeaveRequestResponse])
def pending_requests(
    current_user=Depends(require_roles("manager", "admin")),
    db: Session = Depends(get_db),
):
    query = db.query(LeaveRequest).filter(LeaveRequest.status == "pending")

    if current_user.role == "manager":
        managed_employee_ids = [
            e.id for e in db.query(Employee).filter(
                Employee.manager_id == current_user.id
            ).all()
        ]
        query = query.filter(
            LeaveRequest.employee_id.in_(managed_employee_ids or [-1])
        )

    return query.order_by(LeaveRequest.created_at.desc()).all()


@router.get("/{request_id}", response_model=LeaveRequestResponse)
def get_request(
    request_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = db.query(LeaveRequest).filter(
        LeaveRequest.id == request_id
    ).first()

    if not item:
        raise HTTPException(status_code=404, detail="Leave request not found.")

    if current_user.role == "employee":
        employee = db.query(Employee).filter(
            Employee.user_id == current_user.id
        ).first()

        if not employee or item.employee_id != employee.id:
            raise HTTPException(status_code=403, detail="Access denied.")

    return item


@router.post("/{request_id}/approve", response_model=LeaveRequestResponse)
def approve(
    request_id: int,
    payload: LeaveDecision = LeaveDecision(),
    current_user=Depends(require_roles("manager", "admin")),
    db: Session = Depends(get_db),
):
    check_manager_access(db, current_user, request_id)

    request, error = approve_request(
        db, request_id, current_user.id, payload.comment
    )
    if error:
        raise HTTPException(status_code=400, detail=error)

    return request


@router.post("/{request_id}/reject", response_model=LeaveRequestResponse)
def reject(
    request_id: int,
    payload: LeaveDecision = LeaveDecision(),
    current_user=Depends(require_roles("manager", "admin")),
    db: Session = Depends(get_db),
):
    check_manager_access(db, current_user, request_id)

    request, error = reject_request(
        db, request_id, current_user.id, payload.comment
    )
    if error:
        raise HTTPException(status_code=400, detail=error)

    return request
