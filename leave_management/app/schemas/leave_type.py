from pydantic import BaseModel, Field


class LeaveTypeCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    description: str | None = None


class LeaveTypeUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    description: str | None = None
    is_active: bool | None = None


class LeaveTypeResponse(BaseModel):
    id: int
    name: str
    description: str | None
    is_active: bool

    model_config = {"from_attributes": True}
