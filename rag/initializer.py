"""
Complete RAG Pipeline with Intent-Based Orchestration.

This orchestrator:
1. Classifies user intent using semantic embeddings
2. Routes policy questions to the RAG system
3. Routes database/action operations to service layer
4. Handles leave request workflows
5. Returns consistent responses
"""

import re
from typing import Any, Dict, List, Optional
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from rag.retriever import Retriever
from rag.context_builder import ContextBuilder
from rag.prompt_builder import (
    build_system_prompt,
    build_user_prompt,
    REFUSAL,
)
from rag.llm import GroqLLM
from rag.hallucination_checker import check_hallucination
from rag.intent_classifier import get_intent_classifier

from leave_management.app.models import Employee, LeaveRequest, LeaveType
from leave_management.app.schemas.leave_request import LeaveRequestCreate
from leave_management.app.services.leave_service import (
    approve_request,
    create_leave_request,
    reject_request,
)


class RAGPipeline:
    """
    Main RAG pipeline with orchestration.

    Flow:
        User question with authentication context
              |
              v
        Intent Classification (semantic)
              |
        +-----+-----+-----+-----+-----+-----+
        |           |           |           |
        v           v           v           v
    POLICY_Q    DB_Q        ACTION       WORKFLOW
        |           |           |           |
        v           v           v           v
      RAG       Service      Service     Stateful
       LLM       Layer       Layer      Handler
        |           |           |           |
        +-----+-----+-----+-----+-----+-----+
              |
              v
        Final Response
    """

    def __init__(self):
        print("Initializing Leave Management AI Pipeline...")

        # RAG components for policy questions
        self.retriever = Retriever()
        self.context_builder = ContextBuilder()
        self.llm = GroqLLM()

        # Intent classifier for routing
        self.intent_classifier = get_intent_classifier()

        print("Leave Management AI Pipeline initialized successfully.")

    # ==========================================================
    # MAIN QUERY ENTRY POINT
    # ==========================================================

    def query(
        self,
        question: str,
        db: Optional[Session] = None,
        user_id: Optional[int] = None,
        user_role: Optional[str] = None,
        top_k: int = 5,
        conversation_history=None,
        confirmed: bool = False,
        draft: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute query with intent-based routing."""

        question = (question or "").strip()

        if not question:
            return self._response(
                query=question,
                intent="general",
                answer=REFUSAL,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 10))

        # Always run the intent router so we can use its structured extraction
        # (leave type, dates, reason, request id) as well as the intent itself.
        # The workflow below uses these values instead of trying to re-parse
        # the same user message with regex.
        intent_result = self.intent_classifier.classify(question)
        intent = intent_result.get("intent", "general")
        intent_score = self._safe_float(intent_result.get("score", 0.0), default=0.0)

        # Continue an in-progress manager action.
        if draft and draft.get("manager_action") and user_role in {"manager", "admin"}:
            intent = draft["manager_action"]
            intent_score = 1.0

        # A confirmed employee leave draft must be submitted by the leave
        # workflow even when the classifier labels the short confirmation as
        # `confirm`. The actual submission is still protected by `confirmed`.
        elif draft and confirmed and user_role == "employee":
            intent = "leave_request"
            intent_score = 1.0

        print(f"\n[INTENT] {intent} (score: {intent_score})")

        if intent == "leave_policy":
            return self._handle_policy_question(question=question, top_k=top_k)

        if intent == "leave_balance":
            return self._handle_leave_balance_query(question=question, db=db, user_id=user_id)

        if intent == "my_leaves":
            return self._handle_my_leaves_query(question=question, db=db, user_id=user_id)

        if intent == "leave_status":
            return self._handle_leave_status_query(question=question, db=db, user_id=user_id)

        if intent in {"today_leaves", "pending_leaves", "today_and_pending_leaves"}:
            if user_role not in {"manager", "admin"}:
                return self._response(
                    query=question,
                    intent=intent,
                    answer="You do not have permission to view leave requests.",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )
            return self._handle_manager_query(question=question, intent=intent, db=db, user_id=user_id)

        if intent in {"approve_leave", "reject_leave"}:
            if user_role not in {"manager", "admin"}:
                return self._response(
                    query=question,
                    intent=intent,
                    answer="You do not have permission to approve or reject leave requests.",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )
            return self._handle_manager_action(
                question=question,
                intent=intent,
                db=db,
                user_id=user_id,
                user_role=user_role,
                draft=draft,
            )

        if intent == "leave_request":
            return self._handle_leave_request_workflow(
                question=question,
                db=db,
                user_id=user_id,
                confirmed=confirmed,
                draft=draft,
                intent_result=intent_result,
            )

        return self._handle_policy_question(question=question, top_k=top_k)

    # ==========================================================
    # POLICY QUESTION HANDLER (RAG)
    # ==========================================================

    def _handle_policy_question(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Answer policy questions using RAG.
        """

        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        print("\n[RAG RETRIEVAL]")
        print(f"Retrieved {len(results)} chunks")

        for idx, item in enumerate(results, 1):
            print(
                f"\nResult {idx}: Score={item.get('score'):.4f}, "
                f"Source={item.get('source')}"
            )

        # No relevant context
        if not results:
            print("[RAG] No relevant policy context found.")
            return self._response(
                query=question,
                intent="leave_policy",
                answer=REFUSAL,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        # Build context
        context = self.context_builder.build(results)

        print("\n[CONTEXT TO LLM]")
        print("=" * 70)
        print(context)
        print("=" * 70)

        # Generate answer
        print("\n[LLM] Generating answer...")

        try:
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(question, context)
            answer = self.llm.generate(system_prompt, user_prompt)
        except Exception as exc:
            print(f"[LLM ERROR] {type(exc).__name__}: {exc}")
            return self._response(
                query=question,
                intent="leave_policy",
                answer=(
                    "I retrieved relevant company policy, but the policy "
                    "answering service is temporarily unavailable. Please "
                    "try again shortly."
                ),
                grounded=False,
                hallucination_score=0.0,
                sources=[
                    {
                        "source": item.get("source", "unknown"),
                        "score": self._safe_float(item.get("score", 0.0), default=0.0),
                    }
                    for item in results
                ],
            )

        answer = (answer or "").strip()
        print(f"\n[LLM ANSWER]\n{answer}")

        # Check for hallucination
        try:
            evaluation = check_hallucination(answer, context)
        except Exception as exc:
            print(f"[HALLUCINATION CHECK ERROR] {type(exc).__name__}: {exc}")
            evaluation = {"hallucination_score": 1.0, "grounded": False}

        hallucination_score = self._safe_float(
            evaluation.get("hallucination_score", 1.0),
            default=1.0,
        )

        grounded = bool(evaluation.get("grounded", False))

        print(f"\n[HALLUCINATION CHECK]")
        print(f"Hallucination score: {hallucination_score}")
        print(f"Grounded: {grounded}")

        # Reject ungrounded answers
        if not grounded:
            print("[RAG] Answer rejected as ungrounded.")
            answer = REFUSAL
            hallucination_score = 0.0
            grounded = True

        # Build sources
        sources = [
            {
                "source": item.get("source", "unknown"),
                "score": self._safe_float(item.get("score", 0.0), default=0.0),
            }
            for item in results
        ]

        return self._response(
            query=question,
            intent="leave_policy",
            answer=answer,
            grounded=grounded,
            hallucination_score=hallucination_score,
            sources=sources,
        )

    # ==========================================================
    # DATABASE QUERY HANDLERS
    # ==========================================================

    def _handle_leave_balance_query(
        self,
        question: str,
        db: Optional[Session],
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        """Query employee's leave balance."""

        if not db or not user_id:
            return self._response(
                query=question,
                intent="leave_balance",
                answer="Cannot access leave balance without authentication.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        try:
            from leave_management.app.models import (
                Employee,
                LeaveBalance,
            )

            employee = (
                db.query(Employee)
                .filter(Employee.user_id == user_id)
                .first()
            )

            if not employee:
                return self._response(
                    query=question,
                    intent="leave_balance",
                    answer="Employee profile not found.",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            balances = (
                db.query(LeaveBalance)
                .filter(LeaveBalance.employee_id == employee.id)
                .all()
            )

            if not balances:
                answer = (
                    "You do not have any leave balances configured. "
                    "Please contact Human Resources."
                )
            else:
                lines = ["Your leave balances:"]
                for bal in balances:
                    leave_type = (
                        bal.leave_type.name if bal.leave_type else "Unknown"
                    )
                    available = max(
                        bal.allocated_days - bal.used_days, 0
                    )
                    lines.append(
                        f"- {leave_type}: {available:.1f} days "
                        f"(allocated: {bal.allocated_days:.1f}, used: {bal.used_days:.1f})"
                    )
                answer = "\n".join(lines)

            return self._response(
                query=question,
                intent="leave_balance",
                answer=answer,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        except Exception as exc:
            print(f"[DB QUERY ERROR] {type(exc).__name__}: {exc}")
            raise

    def _handle_my_leaves_query(
        self,
        question: str,
        db: Optional[Session],
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        """Query employee's leave requests."""

        if not db or not user_id:
            return self._response(
                query=question,
                intent="my_leaves",
                answer="Cannot access leave requests without authentication.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        try:
            from leave_management.app.models import (
                Employee,
                LeaveRequest,
            )

            employee = (
                db.query(Employee)
                .filter(Employee.user_id == user_id)
                .first()
            )

            if not employee:
                return self._response(
                    query=question,
                    intent="my_leaves",
                    answer="Employee profile not found.",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            requests_data = (
                db.query(LeaveRequest)
                .filter(LeaveRequest.employee_id == employee.id)
                .order_by(LeaveRequest.created_at.desc())
                .all()
            )

            if not requests_data:
                answer = "You have not submitted any leave requests yet."
            else:
                lines = ["Your leave requests:"]
                for req in requests_data:
                    leave_type = (
                        req.leave_type.name if req.leave_type else "Unknown"
                    )
                    status = req.status.capitalize()
                    lines.append(
                        f"- {leave_type} ({req.start_date} to {req.end_date}): "
                        f"{status}"
                    )
                answer = "\n".join(lines)

            return self._response(
                query=question,
                intent="my_leaves",
                answer=answer,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        except Exception as exc:
            print(f"[DB QUERY ERROR] {type(exc).__name__}: {exc}")
            raise

    def _handle_leave_status_query(
        self,
        question: str,
        db: Optional[Session],
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        """Query latest leave request status."""

        if not db or not user_id:
            return self._response(
                query=question,
                intent="leave_status",
                answer="Cannot access leave status without authentication.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        try:
            from leave_management.app.models import (
                Employee,
                LeaveRequest,
            )

            employee = (
                db.query(Employee)
                .filter(Employee.user_id == user_id)
                .first()
            )

            if not employee:
                return self._response(
                    query=question,
                    intent="leave_status",
                    answer="Employee profile not found.",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            latest = (
                db.query(LeaveRequest)
                .filter(LeaveRequest.employee_id == employee.id)
                .order_by(LeaveRequest.created_at.desc())
                .first()
            )

            if not latest:
                answer = "You have not submitted any leave requests yet."
            else:
                leave_type = (
                    latest.leave_type.name
                    if latest.leave_type
                    else "Unknown"
                )
                status = latest.status.capitalize()
                answer = (
                    f"Your latest leave request is for {leave_type} "
                    f"({latest.start_date} to {latest.end_date}) "
                    f"and is currently **{status}**."
                )

                if latest.status == "pending":
                    answer += (
                        " It is waiting for your manager's review."
                    )
                elif latest.status == "approved":
                    answer += " Your leave has been approved."
                elif latest.status == "rejected":
                    answer += " Your leave request was rejected."
                    if latest.manager_comment:
                        answer += f"\n\nManager's comment: {latest.manager_comment}"

            return self._response(
                query=question,
                intent="leave_status",
                answer=answer,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        except Exception as exc:
            print(f"[DB QUERY ERROR] {type(exc).__name__}: {exc}")
            raise

    # ==========================================================
    # LEAVE REQUEST INITIATION
    # ==========================================================

    def _handle_leave_request_workflow(
        self,
        question: str,
        db: Optional[Session],
        user_id: Optional[int],
        confirmed: bool = False,
        draft: Optional[Dict[str, Any]] = None,
        intent_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create or confirm a leave request from natural-language input."""

        if not db or not user_id:
            return self._response(
                query=question,
                intent="leave_request",
                answer="I can help you apply for leave, but you must be logged in first.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        employee = db.query(Employee).filter(Employee.user_id == user_id).first()
        if not employee:
            return self._response(
                query=question,
                intent="leave_request",
                answer="Employee profile not found.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        active_leave_types = db.query(LeaveType).filter(LeaveType.is_active.is_(True)).all()

        # IMPORTANT: the LLM intent router already extracts structured leave
        # information. Use that information first. The old implementation
        # threw it away and then attempted a second regex-based extraction,
        # which caused valid requests such as:
        #
        #   sick leave 17-09-2026 medical checkup
        #
        # to incorrectly ask for dates again.
        parsed_draft = self._build_leave_request_draft(
            question,
            active_leave_types,
            draft or {},
            extracted=intent_result or {},
        )

        if not parsed_draft.get("leave_type_id"):
            available = ", ".join([lt.name for lt in active_leave_types]) if active_leave_types else "No leave types are available"
            return self._response(
                query=question,
                intent="leave_request",
                answer=(
                    "I can help you apply for leave. Which leave type do you need? "
                    f"Available types: {available}."
                ),
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                draft=parsed_draft,
            )

        if not parsed_draft.get("start_date") or not parsed_draft.get("end_date"):
            return self._response(
                query=question,
                intent="leave_request",
                answer=(
                    "I have the leave type, but I still need the start date and end date "
                    "for the request."
                ),
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                draft=parsed_draft,
            )

        if not parsed_draft.get("reason"):
            return self._response(
                query=question,
                intent="leave_request",
                answer="Please share the reason for the leave request.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                draft=parsed_draft,
            )

        if not confirmed:
            leave_type = db.query(LeaveType).filter(LeaveType.id == parsed_draft["leave_type_id"]).first()
            answer = (
                "Leave Type: " + (leave_type.name if leave_type else parsed_draft.get("leave_type_name", "Unknown")) + "\n"
                f"Start Date: {parsed_draft['start_date']}\n"
                f"End Date: {parsed_draft['end_date']}\n"
                f"Reason: {parsed_draft['reason']}\n\n"
                "Would you like me to submit this leave request?"
            )
            return self._response(
                query=question,
                intent="leave_request",
                answer=answer,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                requires_confirmation=True,
                draft=parsed_draft,
            )

        payload = LeaveRequestCreate(
            leave_type_id=int(parsed_draft["leave_type_id"]),
            start_date=parsed_draft["start_date"],
            end_date=parsed_draft["end_date"],
            reason=parsed_draft["reason"],
        )

        request, error = create_leave_request(db, user_id, payload)
        if error:
            return self._response(
                query=question,
                intent="leave_request",
                answer=f"I could not submit the leave request: {error}",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                draft=parsed_draft,
            )

        created = db.query(LeaveRequest).filter(LeaveRequest.id == request.id).first()
        if not created:
            return self._response(
                query=question,
                intent="leave_request",
                answer="The leave request was submitted but could not be verified in the database.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                request_id=request.id,
                draft=parsed_draft,
            )

        leave_type = created.leave_type.name if created.leave_type else "Unknown"
        answer = (
            "Leave request created successfully.\n\n"
            f"Request ID: {created.id}\n"
            f"Employee: {employee.full_name or employee.user.username}\n"
            f"Leave Type: {leave_type}\n"
            f"Start Date: {created.start_date}\n"
            f"End Date: {created.end_date}\n"
            f"Status: {created.status.upper()}"
        )
        return self._response(
            query=question,
            intent="leave_request",
            answer=answer,
            grounded=True,
            hallucination_score=0.0,
            sources=[],
            request_id=created.id,
            draft={},
        )

    # ==========================================================
    # MANAGER ACTIONS
    # ==========================================================

    def _handle_manager_action(
        self,
        question: str,
        intent: str,
        db: Optional[Session],
        user_id: Optional[int],
        user_role: Optional[str] = None,
        draft: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform actual manager approval/rejection operations."""

        if intent == "pending_leaves":
            return self._handle_pending_leaves_query(question=question, db=db, user_id=user_id)

        request_id = self._extract_request_id(question)
        if not request_id and draft:
            request_id = draft.get("request_id")
        if not request_id:
            return self._response(
                query=question,
                intent=intent,
                answer="I could not find the leave request ID in your message. Please provide the request number.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                draft={"manager_action": intent},
            )

        if not db or not user_id:
            return self._response(
                query=question,
                intent=intent,
                answer="Authentication is required to approve or reject leave requests.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        item = db.query(LeaveRequest).filter(LeaveRequest.id == request_id).first()
        if not item:
            return self._response(
                query=question,
                intent=intent,
                answer=f"Leave request {request_id} was not found.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        employee = db.query(Employee).filter(Employee.id == item.employee_id).first()
        if user_role != "admin":
            if not employee or employee.manager_id != user_id:
                return self._response(
                    query=question,
                    intent=intent,
                    answer="You are not authorized to approve or reject this leave request.",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

        if intent == "approve_leave":
            request, error = approve_request(db, request_id, user_id, comment=None)
            if error:
                return self._response(
                    query=question,
                    intent=intent,
                    answer=f"I could not approve request {request_id}: {error}",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                    request_id=request_id,
                )
            employee = db.query(Employee).filter(Employee.id == request.employee_id).first()
            leave_type = request.leave_type.name if request.leave_type else "Unknown"
            return self._response(
                query=question,
                intent=intent,
                answer=(
                    f"Leave request {request.id} has been approved successfully.\n\n"
                    f"Employee: {employee.full_name if employee else 'Unknown'}\n"
                    f"Leave Type: {leave_type}\n"
                    f"Start Date: {request.start_date}\n"
                    f"End Date: {request.end_date}\n"
                    f"Status: {request.status.upper()}"
                ),
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                request_id=request.id,
            )

        request, error = reject_request(db, request_id, user_id, comment=None)
        if error:
            return self._response(
                query=question,
                intent=intent,
                answer=f"I could not reject request {request_id}: {error}",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
                request_id=request_id,
            )

        employee = db.query(Employee).filter(Employee.id == request.employee_id).first()
        leave_type = request.leave_type.name if request.leave_type else "Unknown"
        return self._response(
            query=question,
            intent=intent,
            answer=(
                f"Leave request {request.id} has been rejected successfully.\n\n"
                f"Employee: {employee.full_name if employee else 'Unknown'}\n"
                f"Leave Type: {leave_type}\n"
                f"Start Date: {request.start_date}\n"
                f"End Date: {request.end_date}\n"
                f"Status: {request.status.upper()}"
            ),
            grounded=True,
            hallucination_score=0.0,
            sources=[],
            request_id=request.id,
        )

    def _handle_pending_leaves_query(
        self,
        question: str,
        db: Optional[Session],
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        """Query pending leave requests (manager view)."""

        if not db or not user_id:
            return self._response(
                query=question,
                intent="pending_leaves",
                answer="Cannot access pending requests without authentication.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        try:
            query = db.query(LeaveRequest).filter(LeaveRequest.status == "pending")

            managed_employee_ids = [
                e.id for e in db.query(Employee).filter(Employee.manager_id == user_id).all()
            ]
            query = query.filter(
                LeaveRequest.employee_id.in_(managed_employee_ids or [-1])
            )

            pending = query.order_by(LeaveRequest.created_at.desc()).all()

            if not pending:
                answer = "No pending leave requests to review."
            else:
                lines = ["Pending Leave Requests:"]
                for req in pending:
                    employee = db.query(Employee).filter(Employee.id == req.employee_id).first()
                    emp_name = employee.full_name if employee else f"Employee {req.employee_id}"
                    leave_type = req.leave_type.name if req.leave_type else "Unknown"
                    lines.append(
                        f"- Request ID: {req.id} | Employee: {emp_name} | "
                        f"Leave Type: {leave_type} | Start: {req.start_date} | "
                        f"End: {req.end_date} | Status: {req.status.upper()}"
                    )
                answer = "\n".join(lines)

            return self._response(
                query=question,
                intent="pending_leaves",
                answer=answer,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        except Exception as exc:
            print(f"[PENDING LEAVES QUERY ERROR] {type(exc).__name__}: {exc}")
            raise

    def _handle_manager_query(
        self,
        question: str,
        intent: str,
        db: Optional[Session],
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        """Handle manager database queries for today's and pending leaves."""

        if not db or not user_id:
            return self._response(
                query=question,
                intent=intent,
                answer="Cannot access leave requests without authentication.",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        today = date.today()
        query = db.query(LeaveRequest)
        managed_employee_ids = [
            e.id for e in db.query(Employee).filter(Employee.manager_id == user_id).all()
        ]

        if managed_employee_ids:
            query = query.filter(LeaveRequest.employee_id.in_(managed_employee_ids))
        else:
            query = query.filter(LeaveRequest.employee_id.in_([-1]))

        if intent == "today_leaves" or intent == "today_and_pending_leaves":
            today_requests = query.filter(
                LeaveRequest.start_date <= today,
                LeaveRequest.end_date >= today,
            ).order_by(LeaveRequest.created_at.desc()).all()
        else:
            today_requests = []

        if intent == "pending_leaves" or intent == "today_and_pending_leaves":
            pending_requests = query.filter(LeaveRequest.status == "pending").order_by(LeaveRequest.created_at.desc()).all()
        else:
            pending_requests = []

        today_lines = ["TODAY'S LEAVE REQUESTS"]
        if not today_requests:
            today_lines.append("No leave requests for today.")
        else:
            for req in today_requests:
                employee = db.query(Employee).filter(Employee.id == req.employee_id).first()
                emp_name = employee.full_name if employee else f"Employee {req.employee_id}"
                leave_type = req.leave_type.name if req.leave_type else "Unknown"
                today_lines.append(
                    f"- Request ID: {req.id} | Employee: {emp_name} | "
                    f"Employee Username: {employee.user.username if employee and employee.user else 'Unknown'} | "
                    f"Leave Type: {leave_type} | Start: {req.start_date} | End: {req.end_date} | "
                    f"Reason: {req.reason} | Status: {req.status.upper()}"
                )

        pending_lines = ["PENDING LEAVE REQUESTS"]
        if not pending_requests:
            pending_lines.append("No pending leave requests.")
        else:
            for req in pending_requests:
                employee = db.query(Employee).filter(Employee.id == req.employee_id).first()
                emp_name = employee.full_name if employee else f"Employee {req.employee_id}"
                leave_type = req.leave_type.name if req.leave_type else "Unknown"
                pending_lines.append(
                    f"- Request ID: {req.id} | Employee: {emp_name} | "
                    f"Employee Username: {employee.user.username if employee and employee.user else 'Unknown'} | "
                    f"Leave Type: {leave_type} | Start: {req.start_date} | End: {req.end_date} | "
                    f"Reason: {req.reason} | Status: {req.status.upper()}"
                )

        if intent == "today_and_pending_leaves":
            answer = "\n\n".join(["\n".join(today_lines), "\n".join(pending_lines)])
        elif intent == "today_leaves":
            answer = "\n".join(today_lines)
        else:
            answer = "\n".join(pending_lines)

        return self._response(
            query=question,
            intent=intent,
            answer=answer,
            grounded=True,
            hallucination_score=0.0,
            sources=[],
        )

    def _build_leave_request_draft(
        self,
        question: str,
        active_leave_types: List[LeaveType],
        existing: Optional[Dict[str, Any]] = None,
        extracted: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        draft = dict(existing or {})
        extracted = extracted or {}
        question_lower = (question or "").lower()

        # ------------------------------------------------------
        # FIRST: use structured values produced by the LLM router
        # ------------------------------------------------------
        # Only copy non-empty values so a follow-up message cannot erase
        # information already collected in the temporary draft.
        extracted_leave_type = extracted.get("leave_type")
        extracted_start = extracted.get("start_date")
        extracted_end = extracted.get("end_date")
        extracted_reason = extracted.get("reason")

        if extracted_leave_type and not draft.get("leave_type_name"):
            draft["leave_type_name"] = str(extracted_leave_type).strip()

        if extracted_start and not draft.get("start_date"):
            parsed_start = self._parse_date_value(str(extracted_start))
            if parsed_start:
                draft["start_date"] = parsed_start

        if extracted_end and not draft.get("end_date"):
            parsed_end = self._parse_date_value(str(extracted_end))
            if parsed_end:
                draft["end_date"] = parsed_end

        if extracted_reason and not draft.get("reason"):
            draft["reason"] = str(extracted_reason).strip()

        if not draft.get("leave_type_name"):
            for leave_type in active_leave_types:
                leave_name = leave_type.name.lower()
                short_name = re.sub(r"\s+leave$", "", leave_name).strip()
                if leave_name in question_lower or re.search(
                    rf"\b{re.escape(short_name)}\b", question_lower
                ):
                    draft["leave_type_name"] = leave_type.name
                    break

        if draft.get("leave_type_name"):
            for leave_type in active_leave_types:
                if leave_type.name.lower() == draft["leave_type_name"].lower():
                    draft["leave_type_id"] = leave_type.id
                    break

        # ------------------------------------------------------
        # FALLBACK PARSING
        # ------------------------------------------------------
        # Keep the old parsers only as a safety net for fields the LLM router
        # did not extract. They are NOT the primary source anymore.
        for candidate in self._extract_date_ranges(question):
            if not draft.get("start_date") and candidate.get("start_date"):
                draft["start_date"] = candidate["start_date"]
            if not draft.get("end_date") and candidate.get("end_date"):
                draft["end_date"] = candidate["end_date"]

        if not draft.get("start_date"):
            start_date = self._parse_relative_date(question_lower, "start")
            if start_date:
                draft["start_date"] = start_date

        # A single supplied date means a ONE-DAY leave unless the user
        # explicitly supplied a different end date. This is the key behavior
        # needed for messages such as:
        #   sick leave 17-09-2026 medical checkup
        if draft.get("start_date") and not draft.get("end_date"):
            default_end = self._parse_relative_date(question_lower, "end")
            if default_end:
                draft["end_date"] = default_end
            else:
                draft["end_date"] = draft["start_date"]

        if not draft.get("reason"):
            reason = self._extract_reason(question)
            if reason:
                draft["reason"] = reason

        # Normalize ISO date strings and protect against an invalid reversed
        # range. A reversed range is treated as a one-day request rather than
        # allowing an invalid payload to reach the service layer.
        if draft.get("start_date"):
            normalized_start = self._parse_date_value(str(draft["start_date"]))
            if normalized_start:
                draft["start_date"] = normalized_start

        if draft.get("end_date"):
            normalized_end = self._parse_date_value(str(draft["end_date"]))
            if normalized_end:
                draft["end_date"] = normalized_end

        if draft.get("start_date") and draft.get("end_date"):
            try:
                start_obj = date.fromisoformat(str(draft["start_date"]))
                end_obj = date.fromisoformat(str(draft["end_date"]))
                if end_obj < start_obj:
                    draft["end_date"] = draft["start_date"]
            except ValueError:
                pass

        return draft

    @staticmethod
    def _extract_request_id(question: str) -> Optional[int]:
        match = re.search(
            r"(?:approve|accept|reject|decline|request|leave request|leave|id)"
            r"\s*(?:id\s*)?[:#-]?\s*(\d+)",
            (question or "").lower(),
            re.I,
        )
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        if re.fullmatch(r"\s*#?\d+\s*", question or ""):
            return int((question or "").strip().lstrip("#"))
        return None

    @staticmethod
    def _parse_date_value(date_text: str) -> Optional[str]:
        if not date_text:
            return None
        text = date_text.strip()
        try:
            return str(date.fromisoformat(text))
        except ValueError:
            pass

        cleaned = text.lower().replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        for date_format in ("%d-%m-%Y", "%d/%m/%Y", "%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(cleaned, date_format).date().isoformat()
            except ValueError:
                continue

        return None

    def _extract_date_ranges(self, question: str) -> List[Dict[str, str]]:
        matches = []
        numeric_dates = re.findall(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", question or "")
        if len(numeric_dates) >= 2:
            start_date = self._parse_date_value(numeric_dates[0])
            end_date = self._parse_date_value(numeric_dates[1])
            if start_date and end_date:
                matches.append({"start_date": start_date, "end_date": end_date})

        patterns = [
            r"(?:from\s+)([A-Za-z0-9\- ]+?)\s+(?:to|until|through)\s+([A-Za-z0-9\- ]+)",
            r"(?:between\s+)([A-Za-z0-9\- ]+?)\s+(?:and)\s+([A-Za-z0-9\- ]+)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, (question or "").lower(), re.I):
                start_text = match.group(1).strip()
                end_text = match.group(2).strip()
                start_date = self._parse_date_value(start_text)
                end_date = self._parse_date_value(end_text)
                if start_date and end_date:
                    matches.append({"start_date": start_date, "end_date": end_date})

        if not matches:
            today = date.today()
            tomorrow = today + timedelta(days=1)
            next_monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
            if "tomorrow" in (question or "").lower():
                matches.append({"start_date": tomorrow.isoformat(), "end_date": tomorrow.isoformat()})
            elif "next monday" in (question or "").lower():
                matches.append({"start_date": next_monday.isoformat(), "end_date": next_monday.isoformat()})
            elif "today" in (question or "").lower():
                matches.append({"start_date": today.isoformat(), "end_date": today.isoformat()})
        return matches

    @staticmethod
    def _parse_relative_date(question: str, kind: str) -> Optional[str]:
        today = date.today()
        if "tomorrow" in question:
            target = today + timedelta(days=1)
            return target.isoformat()
        if "today" in question:
            target = today
            return target.isoformat()
        if "next monday" in question:
            target = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
            return target.isoformat()
        return None

    @staticmethod
    def _extract_reason(question: str) -> Optional[str]:
        lower = (question or "").lower()
        prefix = [
            "because ",
            "as ",
            "due to ",
            "for ",
            "reason ",
            "i am ",
        ]
        for keyword in prefix:
            idx = lower.find(keyword)
            if idx != -1:
                return (question or "")[idx + len(keyword):].strip()
        if "not feeling well" in lower:
            return "Not feeling well"
        if "medical" in lower:
            return "Medical appointment"
        if "personal" in lower:
            return "Personal work"
        return None

    @staticmethod
    def _response(
        query: str,
        intent: str,
        answer: str,
        grounded: bool,
        hallucination_score: float,
        sources: List[Dict[str, Any]],
        request_id: Optional[int] = None,
        requires_confirmation: bool = False,
        draft: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build response matching AIQueryResponse schema."""

        return {
            "query": query,
            "intent": intent,
            "answer": answer,
            "grounded": grounded,
            "hallucination_score": hallucination_score,
            "sources": sources,
            "request_id": request_id,
            "requires_confirmation": requires_confirmation,
            "draft": draft or {},
        }

    # ==========================================================
    # SAFE FLOAT
    # ==========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default


# ==============================================================
# GLOBAL PIPELINE
# ==============================================================

rag_pipeline = RAGPipeline()