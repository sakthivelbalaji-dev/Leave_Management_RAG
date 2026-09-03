from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship
from leave_management.app.database.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False, default="employee")
    is_active = Column(Boolean, nullable=False, default=True)

    employee = relationship(
        "Employee",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        foreign_keys="Employee.user_id",
    )
