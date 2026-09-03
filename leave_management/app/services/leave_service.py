from datetime import datetime, timezone
from sqlalchemy.orm import Session
from leave_management.app.models import Employee, LeaveBalance, LeaveRequest, LeaveType


def calculate_days(start_date, end_date) -> int:
    return (end_date - start_date).days + 1


def get_employee_for_user(db: Session, user_id: int):
    return db.query(Employee).filter(Employee.user_id == user_id).first()


def create_leave_request(db: Session, user_id: int, payload):
    employee = get_employee_for_user(db, user_id)
    if not employee:
        return None, "Employee profile not found."

    leave_type = db.query(LeaveType).filter(
        LeaveType.id == payload.leave_type_id,
        LeaveType.is_active == True,
    ).first()
    if not leave_type:
        return None, "Leave type not found or inactive."

    days = calculate_days(payload.start_date, payload.end_date)

    balance = db.query(LeaveBalance).filter(
        LeaveBalance.employee_id == employee.id,
        LeaveBalance.leave_type_id == payload.leave_type_id,
    ).first()

    if not balance:
        return None, "No leave balance configured for this leave type."

    if balance.available_days < days:
        return None, f"Insufficient leave balance. Available: {balance.available_days:g} days."

    overlap = db.query(LeaveRequest).filter(
        LeaveRequest.employee_id == employee.id,
        LeaveRequest.status.in_(["pending", "approved"]),
        LeaveRequest.start_date <= payload.end_date,
        LeaveRequest.end_date >= payload.start_date,
    ).first()

    if overlap:
        return None, "The requested dates overlap an existing leave request."

    request = LeaveRequest(
        employee_id=employee.id,
        leave_type_id=payload.leave_type_id,
        start_date=payload.start_date,
        end_date=payload.end_date,
        days=days,
        reason=payload.reason,
        status="pending",
    )
    db.add(request)
    db.commit()
    db.refresh(request)
    return request, None


def approve_request(db: Session, request_id: int, reviewer_id: int, comment=None):
    request = db.query(LeaveRequest).filter(LeaveRequest.id == request_id).first()
    if not request:
        return None, "Leave request not found."
    if request.status != "pending":
        return None, f"Request is already {request.status}."

    balance = db.query(LeaveBalance).filter(
        LeaveBalance.employee_id == request.employee_id,
        LeaveBalance.leave_type_id == request.leave_type_id,
    ).first()

    if not balance or balance.available_days < request.days:
        return None, "Insufficient balance to approve this request."

    balance.used_days += request.days
    request.status = "approved"
    request.manager_comment = comment
    request.reviewed_by = reviewer_id
    request.reviewed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(request)
    return request, None


def reject_request(db: Session, request_id: int, reviewer_id: int, comment=None):
    request = db.query(LeaveRequest).filter(LeaveRequest.id == request_id).first()
    if not request:
        return None, "Leave request not found."
    if request.status != "pending":
        return None, f"Request is already {request.status}."

    request.status = "rejected"
    request.manager_comment = comment
    request.reviewed_by = reviewer_id
    request.reviewed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(request)
    return request, None
