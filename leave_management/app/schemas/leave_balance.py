from pydantic import BaseModel, Field


class LeaveBalanceCreate(BaseModel):
    employee_id: int
    leave_type_id: int
    allocated_days: float = Field(ge=0)


class LeaveBalanceUpdate(BaseModel):
    allocated_days: float = Field(ge=0)


class LeaveBalanceResponse(BaseModel):
    id: int
    employee_id: int
    leave_type_id: int
    allocated_days: float
    used_days: float
    available_days: float
