from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from leave_management.app.database.database import Base
from leave_management.app.models import Employee, LeaveBalance, LeaveRequest, LeaveType, User
from rag.initializer import RAGPipeline


def build_test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    manager_user = User(
        username="mgr1",
        email="mgr1@example.com",
        hashed_password="x",
        role="manager",
    )
    employee_user = User(
        username="emp1",
        email="emp1@example.com",
        hashed_password="x",
        role="employee",
    )
    session.add_all([manager_user, employee_user])
    session.commit()

    manager = Employee(user_id=manager_user.id, full_name="Manager One", department="HR", manager_id=None)
    employee = Employee(user_id=employee_user.id, full_name="Employee One", department="IT", manager_id=manager_user.id)
    session.add_all([manager, employee])
    session.commit()

    sick = LeaveType(name="Sick Leave", description="Medical leave", is_active=True)
    casual = LeaveType(name="Casual Leave", description="Casual leave", is_active=True)
    session.add_all([sick, casual])
    session.commit()

    session.add(LeaveBalance(employee_id=employee.id, leave_type_id=sick.id, allocated_days=20, used_days=0))
    session.add(LeaveBalance(employee_id=employee.id, leave_type_id=casual.id, allocated_days=10, used_days=0))
    session.commit()

    return session, manager_user, employee_user, employee, manager, sick, casual


def test_leave_request_creation_requires_confirmation_and_draft():
    session, _, _, employee, _, sick, _ = build_test_db()
    pipeline = RAGPipeline()

    result = pipeline.query(
        "I need sick leave tomorrow because I am not feeling well",
        db=session,
        user_id=employee.user_id,
        user_role="employee",
    )

    assert result["intent"] == "leave_request"
    assert result["requires_confirmation"] is True
    assert result["draft"]["leave_type_name"] == sick.name
    assert result["draft"]["start_date"] == str(date.today() + timedelta(days=1))
    assert "not feeling well" in result["draft"]["reason"].lower()


def test_leave_request_follow_up_keeps_type_and_parses_numeric_date_range():
    session, _, employee_user, _, _, sick, _ = build_test_db()
    pipeline = RAGPipeline()

    first = pipeline.query(
        "sick leave",
        db=session,
        user_id=employee_user.id,
        user_role="employee",
    )
    second = pipeline.query(
        "start: 02-09-2026 to end: 03-09-2026",
        db=session,
        user_id=employee_user.id,
        user_role="employee",
        draft=first["draft"],
    )

    assert second["intent"] == "leave_request"
    assert second["draft"]["leave_type_id"] == sick.id
    assert second["draft"]["start_date"] == "2026-09-02"
    assert second["draft"]["end_date"] == "2026-09-03"
    assert not session.query(LeaveRequest).first()


def test_manager_can_query_today_and_pending_requests():
    session, manager_user, _, _, _, _, _ = build_test_db()
    employee = session.query(Employee).filter(Employee.user_id != manager_user.id).first()
    today = date.today()
    session.add(
        LeaveRequest(
            employee_id=employee.id,
            leave_type_id=1,
            start_date=today,
            end_date=today,
            days=1,
            reason="Medical appointment",
            status="pending",
        )
    )
    session.commit()

    pipeline = RAGPipeline()
    result = pipeline.query(
        "Show me today's and pending leave requests",
        db=session,
        user_id=manager_user.id,
        user_role="manager",
    )

    assert result["intent"] in {"today_and_pending_leaves", "today_leaves", "pending_leaves"}
    assert "TODAY'S LEAVE REQUESTS" in result["answer"].upper()
    assert "PENDING LEAVE REQUESTS" in result["answer"].upper()


def test_manager_approve_leave_updates_database_status():
    session, manager_user, _, employee, _, sick, _ = build_test_db()
    req = LeaveRequest(
        employee_id=employee.id,
        leave_type_id=sick.id,
        start_date=date.today(),
        end_date=date.today(),
        days=1,
        reason="Need rest",
        status="pending",
    )
    session.add(req)
    session.commit()

    pipeline = RAGPipeline()
    result = pipeline.query(
        "Approve request 1",
        db=session,
        user_id=manager_user.id,
        user_role="manager",
    )

    session.refresh(req)
    assert result["request_id"] == req.id
    assert req.status == "approved"
    assert "APPROVED" in result["answer"].upper()


def test_manager_can_approve_request_id_from_follow_up_message():
    session, manager_user, _, employee, _, sick, _ = build_test_db()
    req = LeaveRequest(
        employee_id=employee.id,
        leave_type_id=sick.id,
        start_date=date.today(),
        end_date=date.today(),
        days=1,
        reason="Need rest",
        status="pending",
    )
    session.add(req)
    session.commit()

    pipeline = RAGPipeline()
    pending_action = pipeline.query(
        "approve the leave",
        db=session,
        user_id=manager_user.id,
        user_role="manager",
    )
    result = pipeline.query(
        str(req.id),
        db=session,
        user_id=manager_user.id,
        user_role="manager",
        draft=pending_action["draft"],
    )

    session.refresh(req)
    assert pending_action["draft"]["manager_action"] == "approve_leave"
    assert result["request_id"] == req.id
    assert req.status == "approved"
