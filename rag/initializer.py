from datetime import date, datetime, timedelta
import re
from typing import Any, Dict, List

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
from rag.intent_classifier import SemanticIntentClassifier
from leave_management.app.schemas.leave_request import LeaveRequestCreate
from leave_management.app.services.leave_service import create_leave_request


# ============================================================
# EMBEDDING COMPARISON
# ============================================================

try:
    from rag.comparison_retriever import (
        EmbeddingComparisonRetriever,
    )
except ImportError:
    EmbeddingComparisonRetriever = None


# ============================================================
# DATABASE MODELS
# ============================================================

try:
    from leave_management.app.models.employee import Employee
    from leave_management.app.models.leave_request import LeaveRequest
    from leave_management.app.models.leave_balance import LeaveBalance
    from leave_management.app.models.leave_type import LeaveType

except ImportError:
    from app.models.employee import Employee
    from app.models.leave_request import LeaveRequest
    from app.models.leave_balance import LeaveBalance
    from app.models.leave_type import LeaveType


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:

    def __init__(self):

        print("Initializing Leave Management AI...")

        # ========================================================
        # KNOWLEDGE BASE RAG
        # ========================================================

        self.retriever = Retriever()

        self.context_builder = ContextBuilder()

        self.llm = GroqLLM()
        self.intent_classifier = SemanticIntentClassifier()

        # ========================================================
        # EMBEDDING MODEL COMPARISON
        # ========================================================

        self.comparison_retriever = None

        if EmbeddingComparisonRetriever is not None:

            try:

                print(
                    "\nInitializing embedding comparison..."
                )

                self.comparison_retriever = (
                    EmbeddingComparisonRetriever()
                )

                print(
                    "Embedding comparison initialized."
                )

            except Exception as exc:

                print(
                    "\n[COMPARISON INIT ERROR]"
                )

                print(
                    type(exc).__name__,
                    str(exc),
                )

                print(
                    "Production RAG will continue "
                    "without comparison."
                )

        print(
            "\nLeave Management AI "
            "Pipeline initialized successfully."
        )

    # ==========================================================
    # MAIN QUERY
    # ==========================================================

    def query(
        self,
        question: str,
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:

        question = (
            question or ""
        ).strip()

        if not question:

            return self._response(
                query=question,
                answer=REFUSAL,
                intent="general",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        # ------------------------------------------------------
        # TOP K
        # ------------------------------------------------------

        try:

            top_k = int(top_k)

        except (
            TypeError,
            ValueError,
        ):

            top_k = 5

        top_k = max(
            1,
            min(
                top_k,
                10,
            ),
        )

        # ------------------------------------------------------
        # FASTAPI VALUES
        # ------------------------------------------------------

        db: Session | None = kwargs.get(
            "db"
        )

        # IMPORTANT:
        #
        # current_user.id = users.id
        #
        # It is NOT employees.id.
        #
        user_id = kwargs.get(
            "user_id"
        )

        user_role = (
            kwargs.get(
                "user_role"
            )
            or "employee"
        )

        draft = kwargs.get("draft") or {}
        has_draft = any(value is not None for value in draft.values())
        conversation_history = kwargs.get("conversation_history") or []
        confirmed = bool(kwargs.get("confirmed", False))

        classification = self.intent_classifier.classify(
            question,
            conversation_history=conversation_history,
            current_date=date.today().isoformat(),
            user_role=user_role,
            has_pending_draft=has_draft,
        )

        if classification.get("intent") in {
            "leave_request",
            "confirm",
            "deny",
        }:
            return self._handle_leave_application(
                db=db,
                user_id=user_id,
                question=question,
                classification=classification,
                draft=draft,
                confirmed=confirmed,
            )

        print("\n" + "=" * 70)

        print(
            "[AI QUERY]"
        )

        print(
            "Question:",
            question,
        )

        print(
            "User ID:",
            user_id,
        )

        print(
            "User role:",
            user_role,
        )

        print(
            "Top K:",
            top_k,
        )

        print("=" * 70)

        # ======================================================
        # DETERMINE ROUTE
        # ======================================================

        intent = self._detect_route(
            question
        )

        print(
            "[AI ROUTE]:",
            intent,
        )

        # ======================================================
        # DATABASE
        #
        # These questions NEVER go to the knowledge base.
        # ======================================================

        if intent in (
            "leave_balance",
            "my_leaves",
            "leave_status",
            "my_pending_leaves",
        ):

            return self._handle_employee_database(
                db=db,
                user_id=user_id,
                question=question,
                intent=intent,
            )

        # ======================================================
        # MANAGER / ADMIN DATABASE
        # ======================================================

        if intent == "pending_leaves":

            return self._handle_manager_pending(
                db=db,
                user_id=user_id,
                user_role=user_role,
                question=question,
            )

        # ======================================================
        # KNOWLEDGE BASE POLICY
        # ======================================================

        if intent == "leave_policy":

            return self._handle_policy(
                question=question,
                top_k=top_k,
            )

        # ======================================================
        # GENERAL
        # ======================================================

        return self._handle_general(
            question
        )

    # ==========================================================
    # ROUTING
    # ==========================================================

    def _detect_route(
        self,
        question: str,
    ) -> str:

        q = (
            question
            .lower()
            .strip()
        )

        # ======================================================
        # 1. PERSONAL PENDING LEAVES
        # ======================================================

        personal = any(
            phrase in q
            for phrase in (
                "my pending",
                "my pending leave",
                "my pending leaves",
                "my pending request",
                "my pending requests",
                "pending leave for me",
                "pending leaves for me",
                "pending request for me",
            )
        )

        if personal:

            return "my_pending_leaves"

        # ======================================================
        # 2. LEAVE BALANCE
        # ======================================================

        if any(
            phrase in q
            for phrase in (
                "leave balance",
                "my balance",
                "balance of my leave",
                "how many leaves do i have",
                "how much leave do i have",
                "how many leave days do i have",
                "how many days of leave",
                "remaining leave",
                "leave remaining",
                "available leave",
                "available leaves",
            )
        ):

            return "leave_balance"

        # ======================================================
        # 3. MY LEAVE REQUESTS / HISTORY
        # ======================================================

        if any(
            phrase in q
            for phrase in (
                "show my leave",
                "show my leaves",
                "show my leave requests",
                "show my requests",
                "my leave requests",
                "my leave history",
                "leave history",
                "what leaves did i apply",
                "what leave did i apply",
                "did i apply for leave",
                "leaves i applied",
                "requests i made",
            )
        ):

            return "my_leaves"

        # ======================================================
        # 4. PERSONAL LEAVE STATUS
        # ======================================================

        if any(
            phrase in q
            for phrase in (
                "my leave status",
                "status of my leave",
                "status of my request",
                "has my leave been approved",
                "is my leave approved",
                "is my leave request approved",
                "is my leave request pending",
                "what happened to my leave",
                "latest leave status",
                "latest request status",
            )
        ):

            return "leave_status"

        # ======================================================
        # 5. MANAGER PENDING LEAVES
        # ======================================================

        if any(
            phrase in q
            for phrase in (
                "pending leave requests",
                "pending leaves",
                "pending requests",
                "leave requests waiting for approval",
                "leaves waiting for approval",
                "employees have pending leave",
                "which employees have pending leave",
            )
        ):

            return "pending_leaves"

        # ======================================================
        # 6. COMPANY POLICY / KNOWLEDGE BASE
        # ======================================================

        if any(
            phrase in q
            for phrase in (
                "company policy",
                "leave policy",
                "company leave policy",
                "company working hours",
                "company timing",
                "company timings",
                "working hours",
                "office hours",
                "office timing",
                "office timings",
                "working days",
                "lunch break",
                "attendance policy",
                "hr policy",
                "hr rules",
                "company rules",
                "rules for leave",
                "leave rules",
                "how do i apply for leave",
                "how to apply for leave",
            )
        ):

            return "leave_policy"

        # ======================================================
        # SEMANTIC INTENT CLASSIFIER
        # ======================================================

        try:
            classification = self.intent_classifier.classify(
                question,
            )
            detected = classification.get("intent")
            if detected in (
                "leave_balance",
                "leave_status",
                "my_leaves",
                "leave_policy",
                "pending_leaves",
            ):
                return detected
        except Exception as exc:
            print(
                "[INTENT DETECTION WARNING]",
                type(exc).__name__,
                str(exc),
            )

        return "general"

    # ==========================================================
    # NATURAL-LANGUAGE LEAVE APPLICATIONS
    # ==========================================================

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            return None

        text = value.strip().lower()
        today = date.today()
        if text == "today":
            return today
        if text == "tomorrow":
            return today + timedelta(days=1)
        if text == "yesterday":
            return today - timedelta(days=1)

        weekday_names = {
            name.lower(): index
            for index, name in enumerate((
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ))
        }
        match = re.fullmatch(
            r"(?:next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            text,
        )
        if match:
            target = weekday_names[match.group(1)]
            days_ahead = (target - today.weekday()) % 7
            if days_ahead == 0 or text.startswith("next "):
                days_ahead = days_ahead or 7
            return today + timedelta(days=days_ahead)

        for fmt in (
            "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
            "%B %d, %Y", "%B %d %Y", "%d %B %Y",
        ):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue

        for fmt in ("%B %d", "%d %B", "%b %d", "%d %b"):
            try:
                parsed = datetime.strptime(value.strip(), fmt)
                return parsed.replace(year=today.year).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _type_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower().replace("leave", ""))

    def _resolve_leave_type(self, db: Session, value: Any):
        if not value:
            return None
        requested = self._type_key(str(value))
        leave_types = db.query(LeaveType).filter(LeaveType.is_active == True).all()
        for leave_type in leave_types:
            key = self._type_key(leave_type.name)
            acronym = "".join(
                part[0]
                for part in leave_type.name.split()
                if part.lower() != "leave"
            ).lower()
            if requested in {key, acronym}:
                return leave_type
        return None

    def _handle_leave_application(
        self,
        db: Session | None,
        user_id: int | None,
        question: str,
        classification: Dict[str, Any],
        draft: Dict[str, Any],
        confirmed: bool,
    ) -> Dict[str, Any]:
        if db is None or user_id is None:
            return self._response(
                question, "I cannot apply for leave without authentication.",
                "leave_request", True, 0.0, [],
            )

        intent = classification.get("intent")
        if intent == "deny":
            return self._response(
                question, "Leave application cancelled.", "deny", True, 0.0, [],
            )

        merged = dict(draft)
        for key in ("leave_type", "start_date", "end_date", "reason"):
            value = classification.get(key)
            if value is not None:
                merged[key] = value
        if classification.get("start_date") and not classification.get("end_date"):
            merged["end_date"] = classification["start_date"]

        start = self._parse_date(merged.get("start_date"))
        end = self._parse_date(merged.get("end_date"))
        if start and not end:
            end = start
        if end and not start:
            start = end

        leave_type = self._resolve_leave_type(
            db,
            merged.get("leave_type") or merged.get("leave_type_name"),
        )
        reason = merged.get("reason")
        if start and end and end < start:
            return self._response(
                question,
                "The end date cannot be before the start date. Please provide the corrected dates.",
                "leave_request", True, 0.0, {},
            )

        normalized = {
            "leave_type_id": leave_type.id if leave_type else merged.get("leave_type_id"),
            "leave_type_name": leave_type.name if leave_type else merged.get("leave_type_name") or merged.get("leave_type"),
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "reason": reason,
        }
        missing = []
        if not normalized["leave_type_id"]:
            missing.append("the type of leave")
        if not start:
            missing.append("the date or date range")
        if not reason:
            missing.append("the reason for your leave")

        if missing:
            answer = (
                f"What is {missing[0]}?"
                if len(missing) == 1
                else "Please provide " + " and ".join(missing) + "."
            )
            return self._response(
                question, answer, "leave_request", True, 0.0, [],
            ) | {"draft": normalized}

        if intent == "confirm" or confirmed:
            payload = LeaveRequestCreate(
                leave_type_id=normalized["leave_type_id"],
                start_date=start,
                end_date=end,
                reason=str(reason).strip(),
            )
            request, error = create_leave_request(db, user_id, payload)
            if error:
                return self._response(
                    question,
                    f"I could not submit the leave request.\n\nReason: {error}",
                    "leave_request", True, 0.0, [],
                ) | {"draft": normalized}
            return self._response(
                question,
                f"Leave request submitted successfully. Request ID: {request.id}",
                "leave_request", True, 0.0, [], request_id=request.id,
            )

        answer = (
            "Your leave request is ready:\n\n"
            f"Leave Type: {normalized['leave_type_name']}\n"
            f"Start Date: {normalized['start_date']}\n"
            f"End Date: {normalized['end_date']}\n"
            f"Reason: {normalized['reason']}\n\n"
            "Would you like me to submit it?"
        )
        return self._response(
            question, answer, "leave_request", True, 0.0, [],
        ) | {"requires_confirmation": True, "draft": normalized}

    # ==========================================================
    # EMPLOYEE DATABASE HANDLER
    # ==========================================================

    def _handle_employee_database(
        self,
        db: Session | None,
        user_id: int | None,
        question: str,
        intent: str,
    ) -> Dict[str, Any]:

        if db is None:

            return self._response(
                query=question,
                answer=(
                    "Database access is unavailable."
                ),
                intent=intent,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        if user_id is None:

            return self._response(
                query=question,
                answer=(
                    "I cannot access your leave data "
                    "because you are not authenticated."
                ),
                intent=intent,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        try:

            # ==================================================
            # IMPORTANT FIX
            #
            # current_user.id is users.id.
            #
            # We must find the employee record using:
            #
            # Employee.user_id == current_user.id
            # ==================================================

            employee = (
                db.query(Employee)
                .filter(
                    Employee.user_id
                    == user_id
                )
                .first()
            )

            if employee is None:

                print(
                    "[DATABASE] Employee not found "
                    "for user:",
                    user_id,
                )

                return self._response(
                    query=question,
                    answer=(
                        "Your employee record could not "
                        "be found."
                    ),
                    intent=intent,
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            employee_id = employee.id

            print(
                "[DATABASE] User ID:",
                user_id,
            )

            print(
                "[DATABASE] Employee ID:",
                employee_id,
            )

            # ==================================================
            # LEAVE BALANCE
            # ==================================================

            if intent == "leave_balance":

                balances = (
                    db.query(
                        LeaveBalance
                    )
                    .filter(
                        LeaveBalance.employee_id
                        == employee_id
                    )
                    .all()
                )

                answer = (
                    self._format_leave_balance(
                        employee,
                        balances,
                    )
                )

                return self._response(
                    query=question,
                    answer=answer,
                    intent=intent,
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            # ==================================================
            # MY LEAVES
            # ==================================================

            if intent == "my_leaves":

                leaves = (
                    db.query(
                        LeaveRequest
                    )
                    .filter(
                        LeaveRequest.employee_id
                        == employee_id
                    )
                    .order_by(
                        LeaveRequest.created_at.desc()
                    )
                    .all()
                )

                answer = (
                    self._format_my_leaves(
                        leaves
                    )
                )

                return self._response(
                    query=question,
                    answer=answer,
                    intent=intent,
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            # ==================================================
            # LATEST STATUS
            # ==================================================

            if intent == "leave_status":

                leave = (
                    db.query(
                        LeaveRequest
                    )
                    .filter(
                        LeaveRequest.employee_id
                        == employee_id
                    )
                    .order_by(
                        LeaveRequest.created_at.desc()
                    )
                    .first()
                )

                answer = (
                    self._format_leave_status(
                        leave
                    )
                )

                return self._response(
                    query=question,
                    answer=answer,
                    intent=intent,
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            # ==================================================
            # MY PENDING LEAVES
            # ==================================================

            if intent == "my_pending_leaves":

                leaves = (
                    db.query(
                        LeaveRequest
                    )
                    .filter(
                        LeaveRequest.employee_id
                        == employee_id,
                        LeaveRequest.status
                        == "pending",
                    )
                    .order_by(
                        LeaveRequest.created_at.desc()
                    )
                    .all()
                )

                answer = (
                    self._format_pending_leaves(
                        leaves
                    )
                )

                return self._response(
                    query=question,
                    answer=answer,
                    intent=intent,
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

        except Exception as exc:

            import traceback

            traceback.print_exc()

            return self._response(
                query=question,
                answer=(
                    "I could not retrieve your "
                    "leave information from the database."
                ),
                intent=intent,
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        return self._response(
            query=question,
            answer=(
                "No matching database operation "
                "was found."
            ),
            intent=intent,
            grounded=True,
            hallucination_score=0.0,
            sources=[],
        )

    # ==========================================================
    # MANAGER PENDING LEAVES
    # ==========================================================

    def _handle_manager_pending(
        self,
        db: Session | None,
        user_id: int | None,
        user_role: str,
        question: str,
    ) -> Dict[str, Any]:

        if user_role not in (
            "manager",
            "admin",
        ):

            return self._response(
                query=question,
                answer=(
                    "You do not have permission to "
                    "view employee pending leave requests."
                ),
                intent="pending_leaves",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        if db is None or user_id is None:

            return self._response(
                query=question,
                answer=(
                    "I cannot access leave requests "
                    "without authentication."
                ),
                intent="pending_leaves",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        try:

            managed_employee_ids = [
                employee.id
                for employee in (
                    db.query(Employee)
                    .filter(
                        Employee.manager_id
                        == user_id
                    )
                    .all()
                )
            ]

            if not managed_employee_ids:

                return self._response(
                    query=question,
                    answer=(
                        "There are no employees "
                        "assigned to you."
                    ),
                    intent="pending_leaves",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            leaves = (
                db.query(
                    LeaveRequest
                )
                .filter(
                    LeaveRequest.employee_id.in_(
                        managed_employee_ids
                    ),
                    LeaveRequest.status
                    == "pending",
                )
                .order_by(
                    LeaveRequest.created_at.desc()
                )
                .all()
            )

            answer = (
                self._format_manager_pending(
                    db,
                    leaves,
                )
            )

            return self._response(
                query=question,
                answer=answer,
                intent="pending_leaves",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        except Exception as exc:

            import traceback

            traceback.print_exc()

            return self._response(
                query=question,
                answer=(
                    "I could not retrieve pending "
                    "leave requests."
                ),
                intent="pending_leaves",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

    # ==========================================================
    # POLICY / KNOWLEDGE BASE
    # ==========================================================

    def _handle_policy(
        self,
        question: str,
        top_k: int,
    ) -> Dict[str, Any]:

        print(
            "\n[POLICY]"
        )

        print(
            "Searching knowledge base only..."
        )

        try:

            results = (
                self.retriever.retrieve(
                    query=question,
                    top_k=top_k,
                )
            )

            print(
                "[POLICY] Retrieved:",
                len(results),
                "chunks",
            )

            if not results:

                return self._response(
                    query=question,
                    answer=REFUSAL,
                    intent="leave_policy",
                    grounded=True,
                    hallucination_score=0.0,
                    sources=[],
                )

            context = (
                self.context_builder.build(
                    results
                )
            )

            system_prompt = (
                build_system_prompt()
            )

            user_prompt = (
                build_user_prompt(
                    question,
                    context,
                )
            )

            answer = self.llm.generate(
                system_prompt,
                user_prompt,
            )

            answer = (
                answer or ""
            ).strip()

            print("\nFINAL ANSWER:")
            print(answer or "<EMPTY>")

            # --------------------------------------------------
            # Hallucination check
            # --------------------------------------------------

            try:

                evaluation = (
                    check_hallucination(
                        answer,
                        context,
                    )
                )

            except Exception as exc:

                print(
                    "[HALLUCINATION CHECK ERROR]",
                    type(exc).__name__,
                    str(exc),
                )

                evaluation = {
                    "hallucination_score": 1.0,
                    "grounded": False,
                }

            hallucination_score = (
                self._safe_float(
                    evaluation.get(
                        "hallucination_score",
                        1.0,
                    ),
                    default=1.0,
                )
            )

            grounded = bool(
                evaluation.get(
                    "grounded",
                    False,
                )
            )

            if not grounded:

                answer = REFUSAL

                hallucination_score = 0.0

                grounded = True

            sources = []

            for item in results:

                sources.append(
                    {
                        "source": item.get(
                            "source",
                            "unknown",
                        ),
                        "score": self._safe_float(
                            item.get(
                                "score",
                                0.0,
                            )
                        ),
                    }
                )

            return self._response(
                query=question,
                answer=answer,
                intent="leave_policy",
                grounded=grounded,
                hallucination_score=(
                    hallucination_score
                ),
                sources=sources,
            )

        except Exception as exc:

            import traceback

            traceback.print_exc()

            return self._response(
                query=question,
                answer=(
                    "I could not access the "
                    "leave policy documents."
                ),
                intent="leave_policy",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

    # ==========================================================
    # GENERAL
    # ==========================================================

    def _handle_general(
        self,
        question: str,
    ) -> Dict[str, Any]:

        try:

            answer = self.llm.generate(
                build_system_prompt(),
                build_user_prompt(
                    question,
                    "",
                ),
            )

            return self._response(
                query=question,
                answer=(
                    answer or ""
                ).strip(),
                intent="general",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

        except Exception as exc:

            print(
                "[GENERAL LLM ERROR]",
                type(exc).__name__,
                str(exc),
            )

            return self._response(
                query=question,
                answer=(
                    "I'm sorry, I couldn't "
                    "process your question right now."
                ),
                intent="general",
                grounded=True,
                hallucination_score=0.0,
                sources=[],
            )

    # ==========================================================
    # FORMAT: LEAVE BALANCE
    # ==========================================================

    @staticmethod
    def _format_leave_balance(
        employee,
        balances,
    ) -> str:

        employee_name = (
            getattr(
                employee,
                "full_name",
                None,
            )
            or getattr(
                employee,
                "employee_code",
                None,
            )
            or f"Employee {employee.id}"
        )

        if not balances:

            return (
                f"{employee_name}, "
                "no leave balance records were found."
            )

        lines = [
            f"Leave balance for {employee_name}:",
            "",
        ]

        for balance in balances:

            leave_type = getattr(
                balance,
                "leave_type",
                None,
            )

            leave_name = (
                getattr(
                    leave_type,
                    "name",
                    None,
                )
                or f"Leave Type {balance.leave_type_id}"
            )

            allocated = float(
                balance.allocated_days
                or 0
            )

            used = float(
                balance.used_days
                or 0
            )

            available = max(
                allocated - used,
                0,
            )

            lines.append(
                f"- {leave_name}: "
                f"{available:g} days available "
                f"(Allocated: {allocated:g}, "
                f"Used: {used:g})"
            )

        return "\n".join(
            lines
        )

    # ==========================================================
    # FORMAT: MY LEAVES
    # ==========================================================

    @staticmethod
    def _format_my_leaves(
        leaves,
    ) -> str:

        if not leaves:

            return (
                "You have no leave requests."
            )

        lines = [
            "Your leave requests:",
            "",
        ]

        for leave in leaves:

            leave_type = getattr(
                leave,
                "leave_type",
                None,
            )

            leave_name = (
                getattr(
                    leave_type,
                    "name",
                    None,
                )
                or f"Leave Type {leave.leave_type_id}"
            )

            days = float(
                leave.days
                or 0
            )

            lines.append(
                f"- Leave ID: {leave.id} | "
                f"Type: {leave_name} | "
                f"From: {leave.start_date} | "
                f"To: {leave.end_date} | "
                f"Days: {days:g} | "
                f"Status: "
                f"{str(leave.status).upper()}"
            )

            if leave.reason:

                lines.append(
                    f"  Reason: {leave.reason}"
                )

        return "\n".join(
            lines
        )

    # ==========================================================
    # FORMAT: LATEST STATUS
    # ==========================================================

    @staticmethod
    def _format_leave_status(
        leave,
    ) -> str:

        if leave is None:

            return (
                "You do not have any leave requests."
            )

        status = str(
            leave.status
            or "unknown"
        ).upper()

        if status == "PENDING":

            status_text = (
                "Pending — waiting for "
                "manager approval."
            )

        elif status == "APPROVED":

            status_text = "Approved."

        elif status == "REJECTED":

            status_text = "Rejected."

        else:

            status_text = status

        leave_type = getattr(
            leave,
            "leave_type",
            None,
        )

        leave_name = (
            getattr(
                leave_type,
                "name",
                None,
            )
            or f"Leave Type {leave.leave_type_id}"
        )

        days = float(
            leave.days
            or 0
        )

        return (
            "Your latest leave request:\n\n"
            f"Leave ID: {leave.id}\n"
            f"Leave Type: {leave_name}\n"
            f"From: {leave.start_date}\n"
            f"To: {leave.end_date}\n"
            f"Days: {days:g}\n"
            f"Reason: {leave.reason}\n"
            f"Status: {status_text}"
        )

    # ==========================================================
    # FORMAT: MY PENDING LEAVES
    # ==========================================================

    @staticmethod
    def _format_pending_leaves(
        leaves,
    ) -> str:

        if not leaves:

            return (
                "You have no pending leave requests."
            )

        lines = [
            "Your pending leave requests:",
            "",
        ]

        for leave in leaves:

            leave_type = getattr(
                leave,
                "leave_type",
                None,
            )

            leave_name = (
                getattr(
                    leave_type,
                    "name",
                    None,
                )
                or f"Leave Type {leave.leave_type_id}"
            )

            days = float(
                leave.days
                or 0
            )

            lines.append(
                f"- Leave ID: {leave.id} | "
                f"Type: {leave_name} | "
                f"From: {leave.start_date} | "
                f"To: {leave.end_date} | "
                f"Days: {days:g} | "
                f"Status: PENDING"
            )

        return "\n".join(
            lines
        )

    # ==========================================================
    # FORMAT: MANAGER PENDING
    # ==========================================================

    @staticmethod
    def _format_manager_pending(
        db,
        leaves,
    ) -> str:

        if not leaves:

            return (
                "There are no pending leave requests "
                "from your employees."
            )

        lines = [
            "Pending Leave Requests:",
            "",
        ]

        for leave in leaves:

            employee = (
                db.query(
                    Employee
                )
                .filter(
                    Employee.id
                    == leave.employee_id
                )
                .first()
            )

            employee_name = (
                getattr(
                    employee,
                    "full_name",
                    None,
                )
                or getattr(
                    employee,
                    "employee_code",
                    None,
                )
                or f"Employee {leave.employee_id}"
            )

            leave_type = getattr(
                leave,
                "leave_type",
                None,
            )

            leave_name = (
                getattr(
                    leave_type,
                    "name",
                    None,
                )
                or f"Leave Type {leave.leave_type_id}"
            )

            lines.append(
                f"- Request ID: {leave.id} | "
                f"Employee: {employee_name} | "
                f"Type: {leave_name} | "
                f"From: {leave.start_date} | "
                f"To: {leave.end_date} | "
                f"Status: PENDING"
            )

        return "\n".join(
            lines
        )

    # ==========================================================
    # RESPONSE
    # ==========================================================

    @staticmethod
    def _response(
        query: str,
        answer: str,
        intent: str,
        grounded: bool,
        hallucination_score: float,
        sources: List[
            Dict[str, Any]
        ],
        comparison: Any = None,
        request_id: int | None = None,
    ) -> Dict[str, Any]:

        response = {
            "query": query,
            "answer": answer,
            "intent": intent,
            "grounded": grounded,
            "hallucination_score": (
                hallucination_score
            ),
            "sources": sources,

            # Compatibility with existing schema
            "request_id": request_id,
            "requires_confirmation": False,
            "draft": {},
        }

        if comparison is not None:

            response["comparison"] = (
                comparison
            )

        return response

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


# ============================================================
# GLOBAL PIPELINE
# ============================================================

rag_pipeline = RAGPipeline()