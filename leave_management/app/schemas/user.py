from pydantic import BaseModel


class UserAdminResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    employee_id: int | None = None
    employee_code: str | None = None
    full_name: str | None = None
    department: str | None = None
    manager_id: int | None = None
