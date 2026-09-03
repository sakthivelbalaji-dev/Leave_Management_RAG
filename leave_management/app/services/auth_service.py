from sqlalchemy.orm import Session

from leave_management.app.models import User, Employee
from leave_management.app.core.security import (
    hash_password,
    verify_password,
)


# ============================================================
# ALLOWED ROLES
# ============================================================

ALLOWED_ROLES = {
    "employee",
    "manager",
    "admin",
}


# ============================================================
# GET USER BY ID
# ============================================================

def get_user_by_id(
    db: Session,
    user_id: int,
):
    return (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )


# ============================================================
# GET USER BY USERNAME
# ============================================================

def get_user_by_username(
    db: Session,
    username: str,
):
    return (
        db.query(User)
        .filter(User.username == username)
        .first()
    )


# ============================================================
# AUTHENTICATE USER
# ============================================================

def authenticate_user(
    db: Session,
    username: str,
    password: str,
):
    user = get_user_by_username(
        db,
        username,
    )

    # User does not exist
    if not user:
        return None

    # User account is disabled
    if not user.is_active:
        return None

    # Password is incorrect
    if not verify_password(
        password,
        user.hashed_password,
    ):
        return None

    return user


# ============================================================
# REGISTER USER
# ============================================================

def register_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    role: str = "employee",
    employee_code: str | None = None,
    full_name: str | None = None,
    department: str | None = None,
    manager_id: int | None = None,
):
    # --------------------------------------------------------
    # 1. Normalize role
    # --------------------------------------------------------

    role = role.lower().strip()

    if role not in ALLOWED_ROLES:
        return {
            "success": False,
            "message": (
                "Invalid role. "
                "Allowed roles: employee, manager, admin."
            ),
        }

    # --------------------------------------------------------
    # 2. Check username
    # --------------------------------------------------------

    existing_username = (
        db.query(User)
        .filter(User.username == username)
        .first()
    )

    if existing_username:
        return {
            "success": False,
            "message": "Username already exists.",
        }

    # --------------------------------------------------------
    # 3. Check email
    # --------------------------------------------------------

    existing_email = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_email:
        return {
            "success": False,
            "message": "Email already exists.",
        }

    # --------------------------------------------------------
    # 4. Check employee code
    # --------------------------------------------------------

    if employee_code:

        existing_employee_code = (
            db.query(Employee)
            .filter(
                Employee.employee_code
                == employee_code
            )
            .first()
        )

        if existing_employee_code:
            return {
                "success": False,
                "message": "Employee code already exists.",
            }

    # --------------------------------------------------------
    # 5. Validate manager
    # --------------------------------------------------------

    if manager_id is not None:

        manager = (
            db.query(User)
            .filter(User.id == manager_id)
            .first()
        )

        if not manager:
            return {
                "success": False,
                "message": "Manager user not found.",
            }

        # Only manager-role users can be assigned
        # as managers
        if manager.role != "manager":
            return {
                "success": False,
                "message": (
                    "Selected user is not a manager."
                ),
            }

    # --------------------------------------------------------
    # 6. Create User
    # --------------------------------------------------------

    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role,
        is_active=True,
    )

    db.add(user)

    # Generate user.id before creating employee
    db.flush()

    # --------------------------------------------------------
    # 7. Create Employee
    # --------------------------------------------------------

    employee = Employee(
        user_id=user.id,
        employee_code=employee_code,
        full_name=full_name,
        department=department,
        manager_id=manager_id,
    )

    db.add(employee)

    # --------------------------------------------------------
    # 8. Commit
    # --------------------------------------------------------

    try:

        db.commit()

    except Exception:
        db.rollback()

        return {
            "success": False,
            "message": (
                "Failed to register user. "
                "Please try again."
            ),
        }

    # --------------------------------------------------------
    # 9. Refresh objects
    # --------------------------------------------------------

    db.refresh(user)
    db.refresh(employee)

    # --------------------------------------------------------
    # 10. Return complete result
    # --------------------------------------------------------

    return {
        "success": True,
        "message": "User registered successfully.",
        "user": user,
        "employee": employee,
    }