from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from leave_management.app.database.database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"),
        unique=True, nullable=False
    )
    employee_code = Column(String(50), unique=True, nullable=True, index=True)
    full_name = Column(String(150), nullable=True)
    department = Column(String(150), nullable=True)
    manager_id = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    user = relationship(
        "User", back_populates="employee", foreign_keys=[user_id]
    )
    manager = relationship("User", foreign_keys=[manager_id])
