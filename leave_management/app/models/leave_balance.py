from sqlalchemy import Column, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship
from leave_management.app.database.database import Base


class LeaveBalance(Base):
    __tablename__ = "leave_balances"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(
        Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False
    )
    leave_type_id = Column(
        Integer, ForeignKey("leave_types.id", ondelete="CASCADE"), nullable=False
    )
    allocated_days = Column(Float, nullable=False, default=0)
    used_days = Column(Float, nullable=False, default=0)

    leave_type = relationship("LeaveType", back_populates="balances")

    __table_args__ = (
        UniqueConstraint(
            "employee_id", "leave_type_id",
            name="uq_employee_leave_type",
        ),
    )

    @property
    def available_days(self):
        return max(self.allocated_days - self.used_days, 0)
