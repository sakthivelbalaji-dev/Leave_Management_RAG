"""
LLM-based intent classification for Leave Management AI.

Responsibilities:
    1. Understand the user's natural-language request.
    2. Determine the supported intent.
    3. Extract leave-related information when present.
    4. Handle single-day and multi-day leave correctly.
    5. Never execute database operations.
    6. Never answer policy questions.
    7. Never invent missing information.

The classifier only decides WHAT the user wants.

Actual database operations, RAG retrieval, leave validation,
approval/rejection, and submission must be handled elsewhere.
"""

import json
import re
from typing import Any

from rag.llm import GroqLLM


# ============================================================
# SUPPORTED INTENTS
# ============================================================

SUPPORTED_INTENTS = {
    "leave_policy",
    "leave_request",
    "leave_balance",
    "my_leaves",
    "leave_status",
    "today_leaves",
    "pending_leaves",
    "today_and_pending_leaves",
    "approve_leave",
    "reject_leave",
    "cancel_leave",
    "confirm",
    "deny",
    "general",
}


# ============================================================
# INTENT DEFINITIONS
# ============================================================

INTENT_DEFINITIONS = """
leave_policy:
Questions asking for information that must come from company
policy documents.

Examples of information belonging to this intent:
working hours, working days, attendance rules, leave rules,
leave types, eligibility, leave procedures, holidays, HR policies,
company rules, or other documented company policies.

IMPORTANT:
The classifier only identifies this as a policy question.
The actual answer MUST come from the RAG policy documents.

------------------------------------------------------------

leave_request:
The user wants to create, apply for, or submit a NEW leave request.

This includes requests where the user provides:
- leave type
- date
- date range
- reason
- some combination of these

A request such as:
"I need sick leave"
is still leave_request even if information is missing.

------------------------------------------------------------

leave_balance:
The authenticated user wants to know their own leave balance,
remaining leave, available leave, or used leave.

------------------------------------------------------------

my_leaves:
The authenticated user wants to see their own leave requests,
leave history, submitted applications, or previous leaves.

------------------------------------------------------------

leave_status:
The authenticated user wants to know the status of their own
leave request, including whether it is pending, approved,
rejected, cancelled, or otherwise processed.

------------------------------------------------------------

today_leaves:
A manager or authorized user wants to see leave requests
or employees on leave for today.

------------------------------------------------------------

pending_leaves:
A manager or authorized user wants to see pending leave requests
waiting for approval.

------------------------------------------------------------

today_and_pending_leaves:
A manager or authorized user wants both:
1. today's leave information
2. pending leave requests

------------------------------------------------------------

approve_leave:
A manager or authorized user wants to approve a specific leave request.

------------------------------------------------------------

reject_leave:
A manager or authorized user wants to reject or decline
a specific leave request.

------------------------------------------------------------

cancel_leave:
The user wants to cancel or withdraw an existing leave request.

------------------------------------------------------------

confirm:
The user confirms an action that was previously prepared by the
assistant and is waiting for confirmation.

Examples:
"Yes"
"Yes submit it"
"Confirm"
"Proceed"

IMPORTANT:
"confirm" should only be used as a confirmation when the conversation
contains a previously prepared action/draft.

------------------------------------------------------------

deny:
The user rejects or cancels an action that was previously prepared
by the assistant and is waiting for confirmation.

Examples:
"No"
"Cancel it"
"Don't submit"
"No, don't proceed"

IMPORTANT:
"deny" should only be used when there is a previously prepared
action/draft waiting for confirmation.

------------------------------------------------------------

general:
Greeting, thanks, general help, or a message that does not match
any supported operation.
"""


# ============================================================
# CLASSIFICATION PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are the intent and information extraction engine for a
Leave Management System.

Your job is NOT to answer the user.

Your job is ONLY to:
1. Understand the user's current message.
2. Consider the conversation history.
3. Determine the correct supported intent.
4. Extract information explicitly provided by the user.
5. Return ONLY valid JSON.

SUPPORTED INTENTS
=================
{INTENT_DEFINITIONS}

============================================================
STRICT RULES
============================================================

RULE 1 — NEVER INVENT INFORMATION
----------------------------------
Only extract information explicitly stated by the user or clearly
resolved from the conversation history.

Never invent:
- dates
- leave types
- reasons
- request IDs
- employees
- balances
- policy values
- approval decisions

Missing information must be null.

------------------------------------------------------------

RULE 2 — LEAVE REQUEST DETECTION
---------------------------------
If the user wants to apply, request, create, or submit NEW leave,
the intent is:

"leave_request"

Even if the user has not provided all required information.

For example:

"I need leave"
→ leave_request

"I need sick leave"
→ leave_request

"I want leave next week"
→ leave_request

------------------------------------------------------------

RULE 3 — SINGLE-DAY LEAVE
--------------------------
If the user provides EXACTLY ONE leave date, it means a
SINGLE-DAY leave.

For example:

"I need sick leave on 17-09-2026"

must produce:

start_date = "2026-09-17"
end_date   = "2026-09-17"

Do NOT leave end_date null.

------------------------------------------------------------

RULE 4 — DATE RANGE
--------------------
If the user explicitly provides two dates representing a range:

"I need leave from 17-09-2026 to 19-09-2026"

produce:

start_date = "2026-09-17"
end_date   = "2026-09-19"

------------------------------------------------------------

RULE 5 — ONE DATE MUST NOT REQUIRE ANOTHER DATE
-------------------------------------------------
Never ask for an end date merely because the user supplied one
specific date.

One date means one day.

------------------------------------------------------------

RULE 6 — NATURAL DATE LANGUAGE
-------------------------------
Understand natural date expressions such as:

"tomorrow"
"today"
"next Monday"
"this Friday"
"17 September"
"September 17"
"17-09-2026"

However, do NOT invent a date if it cannot be reliably resolved.

If the application provides today's date, use that date as the
reference for relative expressions.

------------------------------------------------------------

RULE 7 — LEAVE TYPE
--------------------
Extract the leave type only when the user specifies it.

Examples:

"sick leave"
→ "Sick Leave"

"casual leave"
→ "Casual Leave"

"earned leave"
→ "Earned Leave"

Do not invent a leave type.

------------------------------------------------------------

RULE 8 — REASON
----------------
Extract the reason when the user provides one.

Example:

"I need sick leave on 17-09-2026 for a health checkup"

reason:
"health checkup"

If no reason is provided:

reason = null

------------------------------------------------------------

RULE 9 — REQUEST ID
--------------------
For approval or rejection, extract the request ID if explicitly
provided.

Example:

"Approve request 15"

request_id = 15

If no request ID is provided:

request_id = null

Do not invent one.

------------------------------------------------------------

RULE 10 — CONFIRMATION
-----------------------
"Yes submit this leave request"
or
"Yes, submit it"

should be classified as "confirm" ONLY when there is an existing
leave draft/action in the conversation.

If there is no pending action, do not assume what the user wants.

------------------------------------------------------------

RULE 11 — CONVERSATION HISTORY
-------------------------------
Use previous conversation messages to understand references such as:

"yes submit it"
"that leave"
"same date"
"cancel it"
"approve that one"

The conversation history is context only.

It is NOT evidence of company policy.

------------------------------------------------------------

RULE 12 — POLICY QUESTIONS
----------------------------
If the user asks:

"What are the working hours?"
"What is the leave policy?"
"How many sick leaves are allowed?"

classify as:

"leave_policy"

Do NOT answer the question here.

The RAG system must retrieve the answer from the policy document.

------------------------------------------------------------

RULE 13 — DATABASE INFORMATION
--------------------------------
Requests such as:

"What's my leave balance?"
"Show my leaves"
"What is my leave status?"

must be classified appropriately.

The classifier does NOT access or invent database information.

------------------------------------------------------------

RULE 14 — ROLE
---------------
The authenticated user's role is supplied separately.

Do not assume that every user is a manager.

Manager-only operations are:

today_leaves
pending_leaves
today_and_pending_leaves
approve_leave
reject_leave

The application must enforce authorization separately.

------------------------------------------------------------

RULE 15 — OUTPUT
-----------------
Return ONLY this JSON structure:

{{
    "intent": "...",
    "confidence": 0.0,
    "leave_type": null,
    "start_date": null,
    "end_date": null,
    "reason": null,
    "request_id": null
}}

confidence must be a number between 0 and 1.

Do not include markdown.

Do not include explanations.

Do not include additional fields.
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(text: str) -> dict[str, Any]:
    """
    Extract the first JSON object from an LLM response.

    Handles accidental markdown fences such as:

    ```json
    {...}
    ```
    """

    if not text:
        return {}

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Direct JSON
    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    # Find JSON object inside response
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return {}

    try:
        value = json.loads(match.group(0))

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        return {}

    return {}


# ============================================================
# DATE NORMALIZATION
# ============================================================

def _normalize_single_day(
    start_date: Any,
    end_date: Any,
):
    """
    If exactly one date exists, treat it as a single-day leave.

    This is deliberately implemented in Python rather than trusting
    the LLM to do it correctly.
    """

    if start_date and not end_date:
        end_date = start_date

    return start_date, end_date


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def _clean_value(value: Any):
    """
    Convert empty/null-like LLM values to None.
    """

    if value is None:
        return None

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return None

        if value.lower() in {
            "null",
            "none",
            "unknown",
            "not provided",
            "not specified",
        }:
            return None

        return value

    return value


# ============================================================
# INTENT CLASSIFIER
# ============================================================

class SemanticIntentClassifier:

    def __init__(self):
        print("Initializing LLM Intent Classifier...")

        self.llm = GroqLLM()

        print("LLM Intent Classifier initialized.")

    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    def classify(
        self,
        text: str,
        conversation_history=None,
        current_date: str | None = None,
        user_role: str | None = None,
        has_pending_draft: bool = False,
    ):
        """
        Classify the current user message.

        Parameters
        ----------
        text:
            Current user message.

        conversation_history:
            Previous conversation messages.

        current_date:
            Current application date in YYYY-MM-DD format.
            Used to resolve relative dates such as "tomorrow".

        user_role:
            Authenticated user's role.

        has_pending_draft:
            Whether the application currently has an action/draft
            waiting for confirmation.
        """

        text = (text or "").strip()

        if not text:

            return {
                "intent": "general",
                "confidence": 0.0,
                "leave_type": None,
                "start_date": None,
                "end_date": None,
                "reason": None,
                "request_id": None,
            }

        # ----------------------------------------------------
        # BUILD CONVERSATION CONTEXT
        # ----------------------------------------------------

        history_text = ""

        if conversation_history:

            history_lines = []

            for message in conversation_history:

                if isinstance(message, dict):

                    role = message.get(
                        "role",
                        "unknown",
                    )

                    content = message.get(
                        "content",
                        "",
                    )

                else:

                    role = getattr(
                        message,
                        "role",
                        "unknown",
                    )

                    content = getattr(
                        message,
                        "content",
                        "",
                    )

                if content:

                    history_lines.append(
                        f"{role}: {content}"
                    )

            if history_lines:

                history_text = "\n".join(
                    history_lines[-20:]
                )

        # ----------------------------------------------------
        # CURRENT CONTEXT
        # ----------------------------------------------------

        context = f"""
CONVERSATION HISTORY
====================
{history_text if history_text else "No previous conversation."}

CURRENT DATE
============
{current_date if current_date else "Not provided."}

AUTHENTICATED USER ROLE
=======================
{user_role if user_role else "Not provided."}

PENDING ACTION/DRAFT EXISTS
===========================
{"YES" if has_pending_draft else "NO"}

CURRENT USER MESSAGE
====================
{text}
"""

        # ----------------------------------------------------
        # LLM CALL
        # ----------------------------------------------------

        try:

            response = self.llm.generate(
                SYSTEM_PROMPT,
                context,
            )

        except Exception as exc:

            print(
                "[INTENT CLASSIFIER ERROR]",
                type(exc).__name__,
                str(exc),
            )

            return {
                "intent": "general",
                "score": 0.0,
                "confidence": 0.0,
                "leave_type": None,
                "start_date": None,
                "end_date": None,
                "reason": None,
                "request_id": None,
            }

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        result = _extract_json(response)

        if not result:

            print(
                "[INTENT CLASSIFIER] "
                "Invalid JSON returned by LLM."
            )

            return {
                "intent": "general",
                "confidence": 0.0,
                "leave_type": None,
                "start_date": None,
                "end_date": None,
                "reason": None,
                "request_id": None,
            }

        # ----------------------------------------------------
        # INTENT
        # ----------------------------------------------------

        intent = _clean_value(
            result.get("intent")
        )

        if intent not in SUPPORTED_INTENTS:

            intent = "general"

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        try:

            confidence = float(
                result.get(
                    "confidence",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        # ----------------------------------------------------
        # EXTRACTED VALUES
        # ----------------------------------------------------

        leave_type = _clean_value(
            result.get("leave_type")
        )

        start_date = _clean_value(
            result.get("start_date")
        )

        end_date = _clean_value(
            result.get("end_date")
        )

        reason = _clean_value(
            result.get("reason")
        )

        request_id = result.get(
            "request_id"
        )

        # ----------------------------------------------------
        # REQUEST ID NORMALIZATION
        # ----------------------------------------------------

        if request_id is not None:

            try:
                request_id = int(request_id)

            except (
                TypeError,
                ValueError,
            ):
                request_id = None

        # ----------------------------------------------------
        # SINGLE-DAY LEAVE RULE
        # ----------------------------------------------------

        if intent == "leave_request":

            start_date, end_date = (
                _normalize_single_day(
                    start_date,
                    end_date,
                )
            )

        # ----------------------------------------------------
        # CONFIRMATION SAFETY
        # ----------------------------------------------------

        if intent == "confirm" and not has_pending_draft:

            # There is nothing to confirm.
            # Do not allow the model to turn a random "yes"
            # into a database action.
            intent = "general"

        if intent == "deny" and not has_pending_draft:

            intent = "general"

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        final_result = {
            "intent": intent,
            "score": round(
                confidence,
                4,
            ),
            "confidence": round(
                confidence,
                4,
            ),
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date,
            "reason": reason,
            "request_id": request_id,
        }

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        print(
            "\n[INTENT CLASSIFICATION]"
        )

        print(
            "Question:",
            text,
        )

        print(
            "Intent:",
            final_result["intent"],
        )

        print(
            "Confidence:",
            final_result["confidence"],
        )

        print(
            "Leave type:",
            final_result["leave_type"],
        )

        print(
            "Start date:",
            final_result["start_date"],
        )

        print(
            "End date:",
            final_result["end_date"],
        )

        print(
            "Reason:",
            final_result["reason"],
        )

        print(
            "Request ID:",
            final_result["request_id"],
        )

        return final_result


# ============================================================
# SINGLETON
# ============================================================

_classifier = None


def get_intent_classifier():

    global _classifier

    if _classifier is None:

        _classifier = (
            SemanticIntentClassifier()
        )

    return _classifier


# ============================================================
# BACKWARD-COMPATIBLE FUNCTIONS
# ============================================================

def detect_intent(
    text: str,
    conversation_history=None,
    current_date: str | None = None,
    user_role: str | None = None,
    has_pending_draft: bool = False,
) -> str:

    result = get_intent_classifier().classify(
        text=text,
        conversation_history=conversation_history,
        current_date=current_date,
        user_role=user_role,
        has_pending_draft=has_pending_draft,
    )

    return result["intent"]


def classify_intent(
    text: str,
    conversation_history=None,
    current_date: str | None = None,
    user_role: str | None = None,
    has_pending_draft: bool = False,
):

    return get_intent_classifier().classify(
        text=text,
        conversation_history=conversation_history,
        current_date=current_date,
        user_role=user_role,
        has_pending_draft=has_pending_draft,
    )