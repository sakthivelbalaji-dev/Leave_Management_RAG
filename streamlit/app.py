import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Leave Management AI",
    page_icon="📅",
    layout="wide",
)

API_URL = st.sidebar.text_input(
    "API URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# ============================================================
# SESSION STATE
# ============================================================

if "token" not in st.session_state:
    st.session_state.token = None

if "user" not in st.session_state:
    st.session_state.user = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ============================================================
# API HELPERS
# ============================================================

def headers():
    if not st.session_state.token:
        return {}

    return {
        "Authorization": f"Bearer {st.session_state.token}",
        "Content-Type": "application/json",
    }


def api(method, path, **kwargs):
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            headers=headers(),
            timeout=60,
            **kwargs,
        )
        if response.status_code == 401:
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.chat_messages = []
            st.warning("Your session has expired. Please log in again.")
            st.rerun()
        return response
    except requests.RequestException as exc:
        st.error(f"API connection error: {exc}")
        return None


def error_detail(response):
    if response is None:
        return "No response from API."

    try:
        data = response.json()
        return data.get("detail", response.text)
    except Exception:
        return response.text

def get_chat_history():
    """
    Return temporary conversation history for the current
    Streamlit session.

    Only user/assistant text is sent to the AI service.
    UI metadata such as sources and intent is excluded.
    """
    history = []

    for message in st.session_state.chat_messages:
        role = message.get("role")
        content = message.get("content", "").strip()

        if role in {"user", "assistant"} and content:
            history.append({
                "role": role,
                "content": content,
            })

    return history
# ============================================================
# LOGIN / REGISTER
# ============================================================

if not st.session_state.token:

    st.title("📅 Leave Management AI")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")

        username = st.text_input(
            "Username",
            key="login_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button("Login", type="primary"):
            if not username.strip() or not password:
                st.warning("Enter username and password.")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/auth/login",
                        json={
                            "username": username.strip(),
                            "password": password,
                        },
                        timeout=30,
                    )

                    if response.ok:
                        data = response.json()
                        token = data.get("access_token")

                        if not token:
                            st.error("Login succeeded but no access token was returned.")
                        else:
                            st.session_state.token = token

                            me = requests.get(
                                f"{API_URL}/auth/me",
                                headers={
                                    "Authorization": f"Bearer {token}",
                                },
                                timeout=30,
                            )

                            if not me.ok:
                                st.session_state.token = None
                                st.error(
                                    "Login token was received, but /auth/me failed:\n\n"
                                    + error_detail(me)
                                )
                            else:
                                st.session_state.user = me.json()
                                st.rerun()
                    else:
                        st.error(error_detail(response))

                except requests.RequestException as exc:
                    st.error(f"Cannot connect to API: {exc}")

    with tab2:
        st.subheader("Register Employee")

        reg_username = st.text_input(
            "Username",
            key="reg_username",
        )

        reg_email = st.text_input(
            "Email",
            key="reg_email",
        )

        reg_password = st.text_input(
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

        if st.button("Register"):
            try:
                response = requests.post(
                    f"{API_URL}/auth/register",
                    json={
                        "username": reg_username.strip(),
                        "email": reg_email.strip(),
                        "password": reg_password,
                        "role": "employee",
                        "employee_code": employee_code.strip() or None,
                        "full_name": full_name.strip() or None,
                        "department": department.strip() or None,
                    },
                    timeout=30,
                )

                if response.ok:
                    st.success(
                        "Registration successful. Please login."
                    )
                else:
                    st.error(error_detail(response))

            except requests.RequestException as exc:
                st.error(f"Cannot connect to API: {exc}")

    st.stop()


# ============================================================
# CURRENT USER
# ============================================================

user = st.session_state.user or {}
role = user.get("role", "")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.success(
    f"Logged in: {user.get('username', 'user')} ({role})"
)

if st.sidebar.button("Logout"):
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.chat_messages = []
    st.rerun()


# ============================================================
# TABS
# ============================================================

tab_names = [
    "Dashboard",
    "My Leave",
    "AI Assistant",
]

if role in {"manager", "admin"}:
    tab_names.append("Approvals")

if role == "admin":
    tab_names.extend([
        "Leave Types",
        "Balances",
        "Users",
    ])

pages = st.tabs(tab_names)


# ============================================================
# DASHBOARD
# ============================================================

with pages[0]:
    st.header("Dashboard")

    response = api("GET", "/dashboard/me")

    if response and response.ok:
        data = response.json()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Employee",
                data.get("employee", {}).get(
                    "full_name",
                    user.get("username", "-"),
                )
                if isinstance(data.get("employee"), dict)
                else user.get("username", "-"),
            )

        with col2:
            balances = data.get("balances", [])
            st.metric(
                "Leave balance records",
                len(balances) if isinstance(balances, list) else 0,
            )

        with col3:
            requests_data = data.get("leave_requests", [])
            st.metric(
                "Leave requests",
                len(requests_data)
                if isinstance(requests_data, list)
                else 0,
            )

        st.json(data)

    elif response:
        st.error(error_detail(response))


# ============================================================
# MY LEAVE
# ============================================================

with pages[1]:
    st.header("My Leave")

    response = api("GET", "/leave-balances/me")

    if response and response.ok:
        st.subheader("Balances")
        st.dataframe(
            response.json(),
            use_container_width=True,
        )
    elif response:
        st.error(error_detail(response))

    st.subheader("Apply for Leave")

    response = api("GET", "/leave-types")

    type_options = {}

    if response and response.ok:
        for item in response.json():
            if item.get("is_active"):
                type_options[item["name"]] = item["id"]

    if type_options:
        selected = st.selectbox(
            "Leave type",
            list(type_options.keys()),
        )

        start = st.date_input(
            "Start date",
            key="normal_leave_start",
        )

        end = st.date_input(
            "End date",
            key="normal_leave_end",
        )

        reason = st.text_area(
            "Reason",
            key="normal_leave_reason",
        )

        if st.button("Submit Leave Request"):
            response = api(
                "POST",
                "/leave-requests",
                json={
                    "leave_type_id": type_options[selected],
                    "start_date": str(start),
                    "end_date": str(end),
                    "reason": reason,
                },
            )

            if response and response.ok:
                st.success(
                    f"Leave request submitted. "
                    f"Request ID: {response.json().get('id', '-')}"
                )
                st.rerun()

            elif response:
                st.error(error_detail(response))

    else:
        st.info("No active leave types configured.")

    st.subheader("My Requests")

    response = api("GET", "/leave-requests/me")

    if response and response.ok:
        st.dataframe(
            response.json(),
            use_container_width=True,
        )
    elif response:
        st.error(error_detail(response))


# ============================================================
# AI ASSISTANT
# ============================================================

with pages[2]:
    st.header("🤖 Policy AI Assistant")

    st.caption(
        "Ask company-policy questions or apply for leave using natural language."
    )

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if message.get("sources"):
                    with st.expander("RAG sources"):
                        st.json(message["sources"])

                if "intent" in message:
                    st.caption(
                        f"Intent: {message['intent']} | "
                        f"Grounded: {message.get('grounded', True)}"
                    )

    user_message = st.chat_input(
        "Ask about company policy or tell me what leave you need..."
    )

    if user_message:
        question = user_message.strip()

        if not question:
            st.warning("Please enter a message.")
            st.stop()

        st.session_state.chat_messages.append({
            "role": "user",
            "content": question,
        })

        draft = st.session_state.get("pending_leave_draft")
        confirmed = False
        if question.lower().strip() in {"yes", "yes submit it", "confirm", "confirm and submit", "yes i confirm"}:
            confirmed = True

        response = api(
            "POST",
            "/ai/query",
            json={
                "question": question,
                "top_k": 5,
                "confirmed": confirmed,
                "draft": draft,
                "conversation_history": get_chat_history()[-20:],
            },
        )

        if response and response.ok:
            data = response.json()

            if data.get("requires_confirmation"):
                st.session_state.pending_leave_draft = data.get("draft", {})
            elif data.get("draft") is not None:
                st.session_state.pending_leave_draft = data.get("draft", {})
            else:
                st.session_state.pending_leave_draft = {}

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": data.get(
                    "answer",
                    "I could not generate an answer.",
                ),
                "sources": data.get("sources", []),
                "intent": data.get("intent", "unknown"),
                "grounded": data.get("grounded", True),
                "request_id": data.get("request_id"),
            })

        elif response:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "❌ **AI service error**\n\n"
                    f"HTTP status: `{response.status_code}`\n\n"
                    f"`{error_detail(response)}`"
                ),
            })

        else:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "❌ Could not connect to the AI service.",
            })

        st.rerun()

    if st.session_state.get("pending_leave_draft"):
        st.info("A leave draft is ready to submit.")
        if st.button("Yes, submit this leave request"):
            st.session_state.chat_messages.append({
                "role": "user",
                "content": "Yes, submit this leave request",
            })
            response = api(
                "POST",
                "/ai/query",
                json={
                    "question": "Yes, submit this leave request",
                    "top_k": 5,
                    "confirmed": True,
                    "draft": st.session_state.pending_leave_draft,
                    "conversation_history": get_chat_history()[-20:],
                },
            )
            if response and response.ok:
                data = response.json()
                st.session_state.pending_leave_draft = {}
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": data.get("answer", "I could not submit the request."),
                    "sources": data.get("sources", []),
                    "intent": data.get("intent", "unknown"),
                    "grounded": data.get("grounded", True),
                    "request_id": data.get("request_id"),
                })
                st.rerun()
            elif response:
                st.error(error_detail(response))

        if st.button("No, cancel"):
            st.session_state.pending_leave_draft = {}
            st.rerun()


# ============================================================
# APPROVALS
# ============================================================

if role in {"manager", "admin"}:

    with pages[3]:
        st.header("Leave Approvals")

        response = api(
            "GET",
            "/leave-requests/pending",
        )

        if response and response.ok:
            requests_data = response.json()

            if not requests_data:
                st.info("No pending requests.")

            for item in requests_data:
                with st.expander(
                    f"Request #{item['id']} — "
                    f"{item['start_date']} to {item['end_date']}"
                ):
                    st.write(
                        f"Employee ID: {item['employee_id']}"
                    )
                    st.write(
                        f"Days: {item['days']}"
                    )
                    st.write(
                        f"Reason: {item['reason']}"
                    )

                    comment = st.text_area(
                        "Manager comment",
                        key=f"comment_{item['id']}",
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(
                            "Approve",
                            key=f"approve_{item['id']}",
                        ):
                            rr = api(
                                "POST",
                                f"/leave-requests/{item['id']}/approve",
                                json={
                                    "comment": comment,
                                },
                            )

                            if rr and rr.ok:
                                st.success("Approved.")
                                st.rerun()
                            elif rr:
                                st.error(error_detail(rr))

                    with col2:
                        if st.button(
                            "Reject",
                            key=f"reject_{item['id']}",
                        ):
                            rr = api(
                                "POST",
                                f"/leave-requests/{item['id']}/reject",
                                json={
                                    "comment": comment,
                                },
                            )

                            if rr and rr.ok:
                                st.success("Rejected.")
                                st.rerun()
                            elif rr:
                                st.error(error_detail(rr))

        elif response:
            st.error(error_detail(response))


# ============================================================
# ADMIN - LEAVE TYPES
# ============================================================

if role == "admin":

    with pages[4]:
        st.header("Leave Types")

        response = api("GET", "/leave-types")

        if response and response.ok:
            st.dataframe(
                response.json(),
                use_container_width=True,
            )

        name = st.text_input(
            "New leave type name",
            key="new_leave_type_name",
        )

        description = st.text_area(
            "Description",
            key="new_leave_type_description",
        )

        if st.button("Create Leave Type"):
            response = api(
                "POST",
                "/leave-types",
                json={
                    "name": name.strip(),
                    "description": description.strip() or None,
                },
            )

            if response and response.ok:
                st.success("Created.")
                st.rerun()
            elif response:
                st.error(error_detail(response))

    with pages[5]:
        st.header("Balances")
        st.info(
            "Use the admin balance endpoints or Swagger to create/update "
            "employee leave balances."
        )

    with pages[6]:
        st.header("Users")

        response = api("GET", "/users")

        if response and response.ok:
            st.dataframe(
                response.json(),
                use_container_width=True,
            )
        elif response:
            st.error(error_detail(response))
