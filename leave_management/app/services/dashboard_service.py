from sqlalchemy import func
from sqlalchemy.orm import Session
from leave_management.app.models import User, Employee, LeaveBalance, LeaveRequest, LeaveType


def employee_dashboard(db: Session, user_id: int):
    employee = db.query(Employee).filter(Employee.user_id == user_id).first()
    if not employee:
        return None

    balances = db.query(LeaveBalance).filter(
        LeaveBalance.employee_id == employee.id
    ).all()

    requests = db.query(LeaveRequest).filter(
        LeaveRequest.employee_id == employee.id
    ).order_by(LeaveRequest.created_at.desc()).all()

    return {
        "employee": {
            "id": employee.id,
            "employee_code": employee.employee_code,
            "full_name": employee.full_name,
            "department": employee.department,
        },
        "balances": [
            {
                "leave_type_id": b.leave_type_id,
                "leave_type": b.leave_type.name if b.leave_type else None,
                "allocated_days": b.allocated_days,
                "used_days": b.used_days,
                "available_days": b.available_days,
            }
            for b in balances
        ],
        "requests": [
            {
                "id": r.id,
                "leave_type": r.leave_type.name if r.leave_type else None,
                "start_date": str(r.start_date),
                "end_date": str(r.end_date),
                "days": r.days,
                "status": r.status,
                "reason": r.reason,
            }
            for r in requests
        ],
    }


def manager_dashboard(db: Session, user_id: int):
    employee_ids = [
        e.id for e in db.query(Employee).filter(
            Employee.manager_id == user_id
        ).all()
    ]

    pending = db.query(LeaveRequest).filter(
        LeaveRequest.employee_id.in_(employee_ids or [-1]),
        LeaveRequest.status == "pending",
    ).order_by(LeaveRequest.created_at.desc()).all()

    return {
        "managed_employee_count": len(employee_ids),
        "pending_requests": [
            {
                "id": r.id,
                "employee_id": r.employee_id,
                "leave_type": r.leave_type.name if r.leave_type else None,
                "start_date": str(r.start_date),
                "end_date": str(r.end_date),
                "days": r.days,
                "reason": r.reason,
                "status": r.status,
            }
            for r in pending
        ],
    }


def admin_dashboard(db: Session):
    return {
        "users": db.query(func.count(User.id)).scalar() or 0,
        "employees": db.query(func.count(Employee.id)).scalar() or 0,
        "leave_types": db.query(func.count(LeaveType.id)).scalar() or 0,
        "leave_requests": db.query(func.count(LeaveRequest.id)).scalar() or 0,
        "pending_requests": db.query(func.count(LeaveRequest.id)).filter(
            LeaveRequest.status == "pending"
        ).scalar() or 0,
        "approved_requests": db.query(func.count(LeaveRequest.id)).filter(
            LeaveRequest.status == "approved"
        ).scalar() or 0,
        "rejected_requests": db.query(func.count(LeaveRequest.id)).filter(
            LeaveRequest.status == "rejected"
        ).scalar() or 0,
    }
