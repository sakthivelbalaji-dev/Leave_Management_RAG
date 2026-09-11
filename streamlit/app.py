import requests
import streamlit as st
from datetime import datetime, date
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Leave Management AI",
    page_icon="📅",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = st.sidebar.text_input(
    "API URL",
    "http://127.0.0.1:8000",
)


DEFAULT_DRAFT = {
    "leave_type_id": None,
    "leave_type_name": None,
    "start_date": None,
    "end_date": None,
    "reason": None,
}


if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "leave_flow" not in st.session_state:
    st.session_state.leave_flow = False
if "leave_step" not in st.session_state:
    st.session_state.leave_step = None
if "leave_draft" not in st.session_state:
    st.session_state.leave_draft = DEFAULT_DRAFT.copy()
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None
if "evaluation_mode" not in st.session_state:
    st.session_state.evaluation_mode = None


def headers():
    if not st.session_state.token:
        return {}
    return {
        "Authorization": f"Bearer {st.session_state.token}",
        "Content-Type": "application/json",
    }


def api(method, path, **kwargs):
    try:
        return requests.request(
            method,
            f"{API_URL}{path}",
            headers=headers(),
            timeout=300,
            **kwargs,
        )
    except requests.RequestException as exc:
        st.error(f"API connection error: {exc}")
        return None


def parse_date(text):
    text = text.strip()
    for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def get_active_leave_types():
    response = api("GET", "/leave-types")
    if not response or not response.ok:
        return []
    try:
        data = response.json()
        return [item for item in data if item.get("is_active", False)]
    except Exception:
        return []


# ============================================================
# FIND LEAVE TYPE
# ============================================================

def find_leave_type(
    text,
    leave_types,
):

    text_lower = (
        text.lower()
        .strip()
    )

    for item in leave_types:

        name = item["name"].lower()

        if text_lower == name:
            return item

        if name in text_lower:
            return item

    return None


# ============================================================
# DETECT LEAVE APPLICATION
# ============================================================

def is_leave_application_request(text):

    text = text.lower()

    keywords = [
        "apply leave",
        "apply for leave",
        "want to apply leave",
        "want to apply for leave",
        "need to apply leave",
        "need to apply for leave",
        "request leave",
        "request for leave",
        "take leave",
        "want leave",
        "need leave",
        "i need leave",
    ]

    return any(
        keyword in text
        for keyword in keywords
    )


# ============================================================
# RESET LEAVE FLOW
# ============================================================

def reset_leave_flow():

    st.session_state.leave_flow = False

    st.session_state.leave_step = None

    st.session_state.leave_draft = (
        DEFAULT_DRAFT.copy()
    )


# ============================================================
# START LEAVE APPLICATION
# ============================================================

def start_leave_application():

    leave_types = (
        get_active_leave_types()
    )

    if not leave_types:

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                "I cannot start the leave "
                "application because there are "
                "no active leave types configured."
            ),
        })

        return

    st.session_state.leave_flow = True

    st.session_state.leave_draft = (
        DEFAULT_DRAFT.copy()
    )

    st.session_state.leave_step = (
        "leave_type"
    )

    names = ", ".join(
        item["name"]
        for item in leave_types
    )

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": (
            "Sure. I can help you apply "
            "for leave.\n\n"
            f"Available leave types: **{names}**\n\n"
            "Please enter the leave type."
        ),
    })


# ============================================================
# SUBMIT LEAVE REQUEST
# ============================================================

def submit_leave_request():

    draft = (
        st.session_state.leave_draft
    )

    payload = {
        "leave_type_id":
            draft["leave_type_id"],

        "start_date":
            str(draft["start_date"]),

        "end_date":
            str(draft["end_date"]),

        "reason":
            draft["reason"],
    }

    response = api(
        "POST",
        "/leave-requests",
        json=payload,
    )

    if response and response.ok:

        try:

            data = response.json()

            request_id = data.get(
                "id"
            )

        except Exception:

            request_id = None

        if request_id:

            message = (
                "✅ **Leave request "
                "submitted successfully.**\n\n"

                f"Request ID: **{request_id}**\n"

                f"Leave type: "
                f"**{draft['leave_type_name']}**\n"

                f"Start date: "
                f"**{draft['start_date']}**\n"

                f"End date: "
                f"**{draft['end_date']}**\n"

                f"Reason: "
                f"**{draft['reason']}**\n\n"

                "Your request is now "
                "**pending approval**."
            )

        else:

            message = (
                "✅ Leave request submitted "
                "successfully.\n\n"

                "Your request is now "
                "**pending approval**."
            )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": message,
        })

        reset_leave_flow()

        return

    if response:

        try:

            error_data = (
                response.json()
            )

            detail = error_data.get(
                "detail",
                "Failed to submit leave request.",
            )

        except Exception:

            detail = response.text

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                "❌ I could not submit "
                "the leave request.\n\n"
                f"**Reason:** {detail}"
            ),
        })


# ============================================================
# PROCESS LEAVE CONVERSATION
# ============================================================

def process_leave_message(
    user_message,
):

    draft = (
        st.session_state.leave_draft
    )

    leave_types = (
        get_active_leave_types()
    )


    # ========================================================
    # STEP 1 - LEAVE TYPE
    # ========================================================

    if (
        st.session_state.leave_step
        == "leave_type"
    ):

        selected_type = find_leave_type(
            user_message,
            leave_types,
        )

        if not selected_type:

            names = "\n".join(
                f"- {item['name']}"
                for item in leave_types
            )

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "I couldn't identify "
                    "that leave type.\n\n"
                    "Please choose one of these:\n\n"
                    f"{names}"
                ),
            })

            return

        draft["leave_type_id"] = (
            selected_type["id"]
        )

        draft["leave_type_name"] = (
            selected_type["name"]
        )

        st.session_state.leave_step = (
            "start_date"
        )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                f"Leave type selected: "
                f"**{selected_type['name']}**.\n\n"

                "Now enter the **start date**.\n\n"

                "Use format: `YYYY-MM-DD`\n\n"

                "Example: `2026-09-01`"
            ),
        })

        return


    # ========================================================
    # STEP 2 - START DATE
    # ========================================================

    if (
        st.session_state.leave_step
        == "start_date"
    ):

        parsed = parse_date(
            user_message
        )

        if not parsed:

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "Please enter a valid "
                    "start date.\n\n"

                    "Use format: `YYYY-MM-DD`\n\n"

                    "Example: `2026-09-01`"
                ),
            })

            return

        if parsed < date.today():

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "The start date cannot "
                    "be in the past.\n\n"

                    "Please enter another "
                    "start date."
                ),
            })

            return

        draft["start_date"] = parsed

        st.session_state.leave_step = (
            "end_date"
        )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                f"Start date: **{parsed}**.\n\n"

                "Now enter the **end date**.\n\n"

                "Use format: `YYYY-MM-DD`\n\n"

                "For a single-day leave, "
                "enter the same date."
            ),
        })

        return


    # ========================================================
    # STEP 3 - END DATE
    # ========================================================

    if (
        st.session_state.leave_step
        == "end_date"
    ):

        parsed = parse_date(
            user_message
        )

        if not parsed:

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "Please enter a valid "
                    "end date.\n\n"

                    "Use format: `YYYY-MM-DD`"
                ),
            })

            return

        if parsed < draft["start_date"]:

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "The end date cannot "
                    "be before the start date.\n\n"

                    "Please enter another "
                    "end date."
                ),
            })

            return

        draft["end_date"] = parsed

        st.session_state.leave_step = (
            "reason"
        )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                f"End date: **{parsed}**.\n\n"

                "Now enter the **reason "
                "for your leave**."
            ),
        })

        return


    # ========================================================
    # STEP 4 - REASON
    # ========================================================

    if (
        st.session_state.leave_step
        == "reason"
    ):

        reason = (
            user_message.strip()
        )

        if len(reason) < 2:

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "Please provide a "
                    "valid reason for "
                    "your leave."
                ),
            })

            return

        draft["reason"] = reason

        st.session_state.leave_step = (
            "confirmation"
        )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                "Please confirm your "
                "leave request:\n\n"

                f"**Leave type:** "
                f"{draft['leave_type_name']}\n\n"

                f"**Start date:** "
                f"{draft['start_date']}\n\n"

                f"**End date:** "
                f"{draft['end_date']}\n\n"

                f"**Reason:** "
                f"{draft['reason']}\n\n"

                "Type **yes** to submit "
                "or **no** to cancel."
            ),
        })

        return


    # ========================================================
    # STEP 5 - CONFIRMATION
    # ========================================================

    if (
        st.session_state.leave_step
        == "confirmation"
    ):

        answer = (
            user_message
            .lower()
            .strip()
        )

        if answer in {
            "yes",
            "y",
            "confirm",
            "confirmed",
            "submit",
        }:

            submit_leave_request()

            return

        if answer in {
            "no",
            "n",
            "cancel",
            "cancel it",
        }:

            reset_leave_flow()

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "❌ Leave application "
                    "cancelled."
                ),
            })

            return

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                "Please type **yes** to "
                "submit the request or "
                "**no** to cancel it."
            ),
        })


# ============================================================
# LOGIN / REGISTER
# ============================================================

if not st.session_state.token:

    st.title(
        "📅 Leave Management AI"
    )

    tab1, tab2 = st.tabs([
        "Login",
        "Register",
    ])


    # ========================================================
    # LOGIN
    # ========================================================

    with tab1:

        st.subheader(
            "🔐 Login"
        )

        username = st.text_input(
            "Username / Email",
            key="login_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button(
            "Login",
            type="primary",
        ):

            if not username or not password:

                st.warning(
                    "Enter username and password."
                )

            else:

                response = requests.post(
                    f"{API_URL}/auth/login",

                    # IMPORTANT:
                    # JSON, NOT data=
                    json={
                        "username":
                            username,

                        "password":
                            password,
                    },

                    timeout=30,
                )

                if response.ok:

                    data = response.json()

                    st.session_state.token = (
                        data["access_token"]
                    )

                    me = requests.get(
                        f"{API_URL}/auth/me",
                        headers=headers(),
                        timeout=30,
                    )

                    if me.ok:

                        st.session_state.user = (
                            me.json()
                        )

                    st.rerun()

                else:

                    st.error(
                        f"Login failed: "
                        f"{response.text}"
                    )


    # ========================================================
    # REGISTER
    # ========================================================

    with tab2:

        st.subheader(
            "📝 Register Employee"
        )

        username = st.text_input(
            "Username",
            key="reg_username",
        )

        email = st.text_input(
            "Email",
            key="reg_email",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="reg_password",
        )

        employee_code = st.text_input(
            "Employee code",
            key="reg_employee_code",
        )

        full_name = st.text_input(
            "Full name",
            key="reg_full_name",
        )

        department = st.text_input(
            "Department",
            key="reg_department",
        )

        if st.button(
            "Register"
        ):

            response = requests.post(
                f"{API_URL}/auth/register",

                json={
                    "username":
                        username,

                    "email":
                        email,

                    "password":
                        password,

                    "role":
                        "employee",

                    "employee_code":
                        employee_code,

                    "full_name":
                        full_name,

                    "department":
                        department,
                },

                timeout=30,
            )

            if response.ok:

                st.success(
                    "Registration successful. "
                    "Please login."
                )

            else:

                st.error(
                    response.text
                )

    st.stop()


# ============================================================
# LOGGED-IN USER
# ============================================================

user = (
    st.session_state.user
    or {}
)

role = user.get(
    "role",
    "",
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.success(
    f"Logged in: "
    f"{user.get('username', 'user')} "
    f"({role})"
)

if st.sidebar.button(
    "Logout"
):

    st.session_state.token = None

    st.session_state.user = None

    st.session_state.chat_messages = []

    st.session_state.evaluation_result = None

    reset_leave_flow()

    st.rerun()


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "📅 Leave Management AI"
)


# ============================================================
# TABS
# ============================================================

tab_names = [
    "Dashboard",
    "My Leave",
    "AI Assistant",
    "Model Evaluation",
]


if role in {
    "manager",
    "admin",
}:

    tab_names.append(
        "Approvals"
    )


if role == "admin":

    tab_names.extend([
        "Leave Types",
        "Balances",
        "Users",
    ])


pages = st.tabs(
    tab_names
)


# ============================================================
# DASHBOARD
# ============================================================

with pages[0]:

    st.header(
        "Dashboard"
    )

    response = api(
        "GET",
        "/dashboard/me",
    )

    if response and response.ok:

        data = response.json()

        if isinstance(data, dict):

            cols = st.columns(
                len(data)
            )

            for index, (
                key,
                value,
            ) in enumerate(
                data.items()
            ):

                with cols[
                    index
                    % len(cols)
                ]:

                    st.metric(
                        str(key)
                        .replace(
                            "_",
                            " "
                        )
                        .title(),

                        str(value),
                    )

        else:

            st.json(data)

    elif response:

        st.error(
            response.text
        )


# ============================================================
# MY LEAVE
# ============================================================

with pages[1]:

    st.header(
        "My Leave"
    )


    # --------------------------------------------------------
    # BALANCES
    # --------------------------------------------------------

    response = api(
        "GET",
        "/leave-balances/me",
    )

    if response and response.ok:

        st.subheader(
            "Balances"
        )

        st.dataframe(
            response.json(),
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # NORMAL LEAVE FORM
    # --------------------------------------------------------

    st.subheader(
        "Apply for Leave"
    )

    response = api(
        "GET",
        "/leave-types",
    )

    type_options = {}

    if response and response.ok:

        type_options = {
            item["name"]:
                item["id"]

            for item in response.json()

            if item.get(
                "is_active",
                False,
            )
        }


    if type_options:

        selected = st.selectbox(
            "Leave type",
            list(
                type_options.keys()
            ),
        )

        start = st.date_input(
            "Start date"
        )

        end = st.date_input(
            "End date"
        )

        reason = st.text_area(
            "Reason"
        )


        if st.button(
            "Submit Leave Request"
        ):

            response = api(
                "POST",
                "/leave-requests",

                json={
                    "leave_type_id":
                        type_options[
                            selected
                        ],

                    "start_date":
                        str(start),

                    "end_date":
                        str(end),

                    "reason":
                        reason,
                },
            )


            if response and response.ok:

                st.success(
                    "Leave request submitted."
                )

                st.rerun()

            elif response:

                st.error(
                    response.text
                )

    else:

        st.info(
            "No active leave types configured."
        )


    # --------------------------------------------------------
    # MY REQUESTS
    # --------------------------------------------------------

    response = api(
        "GET",
        "/leave-requests/me",
    )

    if response and response.ok:

        st.subheader(
            "My Requests"
        )

        st.dataframe(
            response.json(),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# AI ASSISTANT
# ============================================================

with pages[2]:

    st.header(
        "🤖 Policy AI Assistant"
    )

    st.caption(
        "Ask about company policy or "
        "apply for leave directly through "
        "the chat."
    )


    for message in (
        st.session_state.chat_messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    user_message = st.chat_input(
        "Ask about leave policy or "
        "say 'I want to apply for leave'"
    )


    if user_message:

        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_message,
        })


        response = api(
            "POST",
            "/ai/query",
            json={
                "question": user_message,
                "top_k": 3,
                "draft": st.session_state.leave_draft,
                "conversation_history": st.session_state.chat_messages,
            },
        )

        if response and response.ok:
            data = response.json()
            answer = data.get(
                "answer",
                "I could not generate an answer.",
            )
            returned_draft = data.get("draft") or {}
            if data.get("intent") == "leave_request" and returned_draft:
                st.session_state.leave_draft = returned_draft
                st.session_state.leave_flow = data.get(
                    "requires_confirmation", False,
                )
            elif not data.get("requires_confirmation"):
                reset_leave_flow()

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
            })
        elif response:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "❌ AI service error:\n\n"
                    f"{response.text}"
                ),
            })

        st.rerun()


# ============================================================
# MODEL EVALUATION
# ============================================================

with pages[3]:

    st.header(
        "📊 Embedding Model Evaluation"
    )

    st.markdown(
        """
        Compare **Qwen3-Embedding-0.6B** and
        **BAAI/bge-small-en-v1.5** on the same
        Leave Management knowledge base.

        You can enter **any wording** related
        to the documents.
        """
    )

    st.divider()


    # ========================================================
    # QUERY
    # ========================================================

    evaluation_question = st.text_area(
        "🔎 Evaluation Question",

        placeholder="Enter any question about the available leave-management documents.",

        height=110,

        key="evaluation_question",
    )


    # ========================================================
    # TOP K
    # ========================================================

    evaluation_top_k = st.slider(
        "Top K",

        min_value=1,

        max_value=10,

        value=5,

        key="evaluation_top_k",
    )


    # ========================================================
    # COMPARE BUTTON
    # ========================================================

    if st.button(
        "🚀 Compare Models",
        type="primary",
        use_container_width=True,
    ):

        question = (
            evaluation_question
            .strip()
        )


        if not question:

            st.warning(
                "Please enter an evaluation question."
            )

        else:
                question = evaluation_question.strip()
                if not question:
                    st.warning("Please enter an evaluation question.")
                else:
                    with st.spinner("Running QWEN and BAAI..."):
                        response = api(
                            "POST",
                            "/ai/evaluate",
                            json={
                                "question": question,
                                "top_k": evaluation_top_k,
                            },
                        )
                    st.session_state.evaluation_mode = "free_query"
                    if response and response.ok:
                        st.session_state.evaluation_result = response.json()
                        st.success("Model comparison completed.")
                    elif response:
                        try:
                            error_message = response.json().get(
                                "detail", "Unknown evaluation error."
                            )
                        except Exception:
                            error_message = response.text
                        st.error(
                            f"Evaluation failed ({response.status_code}): {error_message}"
                        )
        result = st.session_state.evaluation_result or {}
        qwen = result.get("qwen", {})
        BAAI = result.get("BAAI", {})

        qwen_score = float(
            qwen.get("top_score", 0.0)
        )

        BAAI_score = float(
            BAAI.get("top_score", 0.0)
        )   

        # ====================================================

        st.subheader(
            "🤖 Models"
        )

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "QWEN",
                "Qwen3-Embedding-0.6B",
            )


        with col2:

            st.metric(
                "BAAI",
                "BAAI/bge-small-en-v1.5",
            )


        # ====================================================
        # TOP SIMILARITY
        # ====================================================

        st.subheader(
            "🎯 Similarity Score"
        )

        similarity_col1, similarity_col2 = (
            st.columns(2)
        )


        qwen_score = float(
            qwen.get(
                "top_score",
                0.0,
            )
        )

        BAAI_score = float(
            BAAI.get(
                "top_score",
                0.0,
            )
        )


        with similarity_col1:

            st.metric(
                "QWEN Top Score",
                f"{qwen_score:.4f}",
            )


        with similarity_col2:

            st.metric(
                "BAAI Top Score",
                f"{BAAI_score:.4f}",
            )


        if qwen_score > BAAI_score:

            st.success(
                "🏆 QWEN has the higher "
                "top similarity score."
            )

        elif BAAI_score > qwen_score:

            st.success(
                "🏆 BAAI has the higher "
                "top similarity score."
            )

        else:

            st.info(
                "🤝 Both models have the "
                "same top similarity score."
            )


        st.subheader(
            "Retrieval Metrics"
        )

        metrics_df = pd.DataFrame([
            {
                "Metric": "Top Similarity",
                "QWEN": qwen.get("top_score", 0.0),
                "BAAI": BAAI.get("top_score", 0.0),
            },
            {
                "Metric": "Average Similarity @K",
                "QWEN": qwen.get("average_score", 0.0),
                "BAAI": BAAI.get("average_score", 0.0),
            },
            {
                "Metric": "Precision",
                "QWEN": qwen.get("precision", "N/A"),
                "BAAI": BAAI.get("precision", "N/A"),
            },
            {
                "Metric": "Recall",
                "QWEN": qwen.get("recall", "N/A"),
                "BAAI": BAAI.get("recall", "N/A"),
            },
            {
                "Metric": "F1 Score",
                "QWEN": qwen.get("f1", "N/A"),
                "BAAI": BAAI.get("f1", "N/A"),
            },
            {
                "Metric": "MRR",
                "QWEN": qwen.get("mrr", "N/A"),
                "BAAI": BAAI.get("mrr", "N/A"),
            },
            {
                "Metric": "Embedding Latency (ms)",
                "QWEN": qwen.get("embedding_latency_ms", 0.0),
                "BAAI": BAAI.get("embedding_latency_ms", 0.0),
            },
            {
                "Metric": "Retrieval Latency (ms)",
                "QWEN": qwen.get("retrieval_latency_ms", 0.0),
                "BAAI": BAAI.get("retrieval_latency_ms", 0.0),
            },
            {
                "Metric": "Total Latency (ms)",
                "QWEN": qwen.get("total_latency_ms", 0.0),
                "BAAI": BAAI.get("total_latency_ms", 0.0),
            },
        ])

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # LATENCY
        # ====================================================

        st.subheader(
            "⚡ Latency Comparison"
        )


        latency_rows = [

            {
                "Metric":
                    "Embedding Latency (ms)",

                "QWEN":
                    qwen.get(
                        "embedding_latency_ms",
                        0.0,
                    ),

                "BAAI":
                    BAAI.get(
                        "embedding_latency_ms",
                        0.0,
                    ),
            },

            {
                "Metric":
                    "Retrieval Latency (ms)",

                "QWEN":
                    qwen.get(
                        "retrieval_latency_ms",
                        0.0,
                    ),

                "BAAI":
                    BAAI.get(
                        "retrieval_latency_ms",
                        0.0,
                    ),
            },

            {
                "Metric":
                    "Total Latency (ms)",

                "QWEN":
                    qwen.get(
                        "total_latency_ms",
                        0.0,
                    ),

                "BAAI":
                    BAAI.get(
                        "total_latency_ms",
                        0.0,
                    ),
            },
        ]


        latency_df = pd.DataFrame(
            latency_rows
        )


        st.dataframe(
            latency_df,
            use_container_width=True,
            hide_index=True,
        )


        # ====================================================
        # OVERALL COMPARISON
        # ====================================================

        comparison = result.get(
            "comparison",
            {},
        )


        if comparison:

            st.subheader(
                "🏆 Model Comparison"
            )


            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.metric(
                    "Similarity Winner",
                    comparison.get(
                        "similarity_winner",
                        "N/A",
                    ),
                )


            with c2:

                st.metric(
                    "Speed Winner",
                    comparison.get(
                        "speed_winner",
                        "N/A",
                    ),
                )


            with c3:

                agreement = comparison.get(
                    "top_result_agreement",
                    False,
                )

                st.metric(
                    "Top Result Agreement",
                    "Yes"
                    if agreement
                    else "No",
                )


            with c4:

                overlap = comparison.get(
                    "result_overlap_percentage",
                    0.0,
                )

                st.metric(
                    "Result Overlap",
                    f"{float(overlap):.2f}%",
                )

            if comparison.get("evaluation_available"):
                st.subheader("Evaluation Metrics")
                evaluation_df = pd.DataFrame([
                    {
                        "Metric": "Precision",
                        "QWEN": qwen.get("precision", 0.0),
                        "BAAI": BAAI.get("precision", 0.0),
                    },
                    {
                        "Metric": "Recall",
                        "QWEN": qwen.get("recall", 0.0),
                        "BAAI": BAAI.get("recall", 0.0),
                    },
                    {
                        "Metric": "F1 Score",
                        "QWEN": qwen.get("f1", 0.0),
                        "BAAI": BAAI.get("f1", 0.0),
                    },
                    {
                        "Metric": "MRR",
                        "QWEN": qwen.get("mrr", 0.0),
                        "BAAI": BAAI.get("mrr", 0.0),
                    },
                ])
                st.dataframe(
                    evaluation_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Precision, recall, F1, and MRR are available for questions in the evaluation dataset."
                )


        # ====================================================
        # RETRIEVED RESULTS
        # ====================================================

        st.subheader(
            f"🔍 Top {evaluation_top_k} Retrieved Results"
        )


        result_col1, result_col2 = (
            st.columns(2)
        )


        # ====================================================
        # QWEN
        # ====================================================

        with result_col1:

            st.markdown(
                "### 🔵 Qwen3-Embedding-0.6B"
            )

            qwen_results = (
                qwen.get(
                    "results",
                    [],
                )
            )


            if not qwen_results:

                st.warning(
                    "QWEN returned no results."
                )

            else:

                for index, item in enumerate(
                    qwen_results,
                    start=1,
                ):

                    score = float(
                        item.get(
                            "score",
                            0.0,
                        )
                    )

                    source = item.get(
                        "source",
                        "unknown",
                    )

                    chunk_id = item.get(
                        "chunk_id",
                        "unknown",
                    )

                    content = item.get(
                        "content",
                        "",
                    )


                    with st.expander(
                        f"#{index} | "
                        f"Score: {score:.4f}"
                    ):

                        st.write(
                            f"**Chunk:** {chunk_id}"
                        )

                        st.write(
                            f"**Source:** {source}"
                        )

                        st.write(
                            content
                        )


        # ====================================================
        # BAAI
        # ====================================================

        with result_col2:

            st.markdown(
                "### 🟢 BAAI-Embedding-0.6B"
            )

            BAAI_results = (
                BAAI.get(
                    "results",
                    [],
                )
            )


            if not BAAI_results:

                st.warning(
                    "BAAI returned no results."
                )

            else:

                for index, item in enumerate(
                    BAAI_results,
                    start=1,
                ):

                    score = float(
                        item.get(
                            "score",
                            0.0,
                        )
                    )

                    source = item.get(
                        "source",
                        "unknown",
                    )

                    chunk_id = item.get(
                        "chunk_id",
                        "unknown",
                    )

                    content = item.get(
                        "content",
                        "",
                    )


                    with st.expander(
                        f"#{index} | "
                        f"Score: {score:.4f}"
                    ):

                        st.write(
                            f"**Chunk:** {chunk_id}"
                        )

                        st.write(
                            f"**Source:** {source}"
                        )

                        st.write(
                            content
                        )


        # ====================================================
        # OVERALL RECOMMENDATION
        # ====================================================

        st.divider()

        st.subheader(
            "💡 Recommendation"
        )


        qwen_latency = float(
            qwen.get(
                "total_latency_ms",
                0.0,
            )
        )

        BAAI_latency = float(
            BAAI.get(
                "total_latency_ms",
                0.0,
            )
        )


        if qwen_score > BAAI_score:

                st.info(
                    "QWEN has the higher top similarity for this query."
                )

        elif BAAI_score > qwen_score:

                st.info(
                    "BAAI has the higher top similarity for this query."
                )

        else:

                st.info(
                    "Both models produced the "
                    "same top similarity score."
                )


        st.caption(
            f"Total latency — "
            f"QWEN: {qwen_latency:.2f} ms | "
            f"BAAI: {BAAI_latency:.2f} ms"
        )


# ============================================================
# APPROVALS
# ============================================================

approval_page_index = 4


if role in {
    "manager",
    "admin",
}:

    with pages[approval_page_index]:

        st.header(
            "Leave Approvals"
        )

        response = api(
            "GET",
            "/leave-requests/pending",
        )


        if response and response.ok:

            requests_data = (
                response.json()
            )


            if not requests_data:

                st.info(
                    "No pending requests."
                )


            for item in requests_data:

                with st.expander(
                    f"Request #{item['id']} — "
                    f"{item['start_date']} to "
                    f"{item['end_date']}"
                ):

                    st.write(
                        f"Employee ID: "
                        f"{item['employee_id']}"
                    )

                    st.write(
                        f"Days: "
                        f"{item['days']}"
                    )

                    st.write(
                        f"Reason: "
                        f"{item['reason']}"
                    )


                    comment = st.text_area(
                        "Manager comment",

                        key=(
                            f"comment_"
                            f"{item['id']}"
                        ),
                    )


                    col1, col2 = (
                        st.columns(2)
                    )


                    # ----------------------------------------
                    # APPROVE
                    # ----------------------------------------

                    with col1:

                        if st.button(
                            "Approve",

                            key=(
                                f"approve_"
                                f"{item['id']}"
                            ),
                        ):

                            rr = api(
                                "POST",

                                f"/leave-requests/"
                                f"{item['id']}/approve",

                                json={
                                    "comment":
                                        comment
                                },
                            )


                            if rr and rr.ok:

                                st.success(
                                    "Approved."
                                )

                                st.rerun()

                            elif rr:

                                st.error(
                                    rr.text
                                )


            response = api(
                "POST",
                "/leave-types",

                json={
                    "name":
                        name,

                    "description":
                        description,
                },
            )


            if response and response.ok:

                st.success(
                    "Created."
                )

                st.rerun()

            elif response:

                st.error(
                    response.text
                )


    # ========================================================
    # BALANCES
    # ========================================================

    with pages[6]:

        st.header(
            "Balances"
        )

        st.info(
            "Use Swagger to create and "
            "update employee leave balances."
        )


    # ========================================================
    # USERS
    # ========================================================

    with pages[7]:

        st.header(
            "Users"
        )

        response = api(
            "GET",
            "/users",
        )


        if response and response.ok:

            st.dataframe(
                response.json(),
                use_container_width=True,
                hide_index=True,
            )