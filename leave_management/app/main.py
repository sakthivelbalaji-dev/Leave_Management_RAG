from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from leave_management.app.database.database import Base, engine
from leave_management.app.models import (
    User, Employee, LeaveType, LeaveBalance, LeaveRequest
)
from leave_management.app.routers.auth import router as auth_router
from leave_management.app.routers.users import router as users_router
from leave_management.app.routers.leave_types import router as leave_types_router
from leave_management.app.routers.leave_balances import router as leave_balances_router
from leave_management.app.routers.leave_requests import router as leave_requests_router
from leave_management.app.routers.dashboard import router as dashboard_router
from leave_management.app.routers.ai import router as ai_router

# MVP database initialization. Use Alembic migrations for production.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Knackforge Leave Management AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(leave_types_router)
app.include_router(leave_balances_router)
app.include_router(leave_requests_router)
app.include_router(dashboard_router)
app.include_router(ai_router)


@app.get("/")
def root():
    return {"success": True, "message": "Leave Management API is running"}


@app.get("/health")
def health():
    return {"success": True, "status": "healthy"}
