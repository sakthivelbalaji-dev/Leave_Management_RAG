from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from sqlalchemy.orm import Session

from leave_management.app.database.database import get_db

from leave_management.app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

from leave_management.app.services.auth_service import (
    authenticate_user,
    get_user_by_id,
    register_user,
)

from leave_management.app.core.security import (
    create_access_token,
    decode_access_token,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# HTTP BEARER
# ============================================================

security = HTTPBearer()


# ============================================================
# REGISTER
# ============================================================

@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):

    result = register_user(
        db=db,
        username=request.username,
        email=str(request.email),
        password=request.password,
        role=request.role,
        employee_code=request.employee_code,
        full_name=request.full_name,
        department=request.department,
        manager_id=request.manager_id,
    )

    # Registration failed
    if not result["success"]:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"],
        )

    # Get objects returned by service
    user = result["user"]
    employee = result.get("employee")

    # --------------------------------------------------------
    # Registration response
    # --------------------------------------------------------

    return {
        "success": True,
        "message": result["message"],

        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active,
        },

        "employee": {
            "id": (
                employee.id
                if employee
                else None
            ),

            "employee_code": (
                employee.employee_code
                if employee
                else None
            ),

            "full_name": (
                employee.full_name
                if employee
                else None
            ),

            "department": (
                employee.department
                if employee
                else None
            ),

            "manager_id": (
                employee.manager_id
                if employee
                else None
            ),
        },
    }


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):

    # --------------------------------------------------------
    # Authenticate
    # --------------------------------------------------------

    user = authenticate_user(
        db=db,
        username=request.username,
        password=request.password,
    )

    if user is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    # --------------------------------------------------------
    # Create JWT
    # --------------------------------------------------------

    token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
    )

    # --------------------------------------------------------
    # Return token
    # --------------------------------------------------------

    return {
        "access_token": token,
        "token_type": "bearer",
    }


# ============================================================
# CURRENT USER
# ============================================================

@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    credentials: HTTPAuthorizationCredentials = Depends(
        security
    ),
    db: Session = Depends(get_db),
):

    # --------------------------------------------------------
    # Get token
    # --------------------------------------------------------

    token = credentials.credentials

    # --------------------------------------------------------
    # Decode token
    # --------------------------------------------------------

    payload = decode_access_token(token)

    if payload is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    # --------------------------------------------------------
    # Get user ID
    # --------------------------------------------------------

    user_id = payload.get("sub")

    if user_id is None:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )

    # --------------------------------------------------------
    # Convert user ID
    # --------------------------------------------------------

    try:

        user_id = int(user_id)

    except (TypeError, ValueError):

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )

    # --------------------------------------------------------
    # Get user
    # --------------------------------------------------------

    user = get_user_by_id(
        db=db,
        user_id=user_id,
    )

    if user is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    # --------------------------------------------------------
    # Check active account
    # --------------------------------------------------------

    if not user.is_active:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive.",
        )

    return user