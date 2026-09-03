import requests
import streamlit as st

API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Leave Management AI",
    page_icon="📅",
    layout="wide",
)

st.title(" Leave Management AI")

if "token" not in st.session_state:
    st.session_state.token = None

if "user" not in st.session_state:
    st.session_state.user = None


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
            timeout=60,
            **kwargs,
        )
    except requests.RequestException as exc:
        st.error(f"API connection error: {exc}")
        return None


if not st.session_state.token:
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button("Login", type="primary"):
            response = requests.post(
                f"{API_URL}/auth/login",
                json={
                    "username": username,
                    "password": password,
                },
                timeout=30,
            )

            if response.ok:
                st.session_state.token = response.json()["access_token"]

                me = requests.get(
                    f"{API_URL}/auth/me",
                    headers=headers(),
                    timeout=30,
                )

                if me.ok:
                    st.session_state.user = me.json()

                st.rerun()
            else:
                st.error(response.text)

    with tab2:
        st.subheader("Register Employee")
        username = st.text_input("Username", key="reg_username")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input(
            "Password",
            type="password",
            key="reg_password",
        )
        employee_code = st.text_input(
            "Employee code",
            key="reg_employee_code",
        )
        full_name = st.text_input("Full name", key="reg_full_name")
        department = st.text_input("Department", key="reg_department")

        if st.button("Register"):
            response = requests.post(
                f"{API_URL}/auth/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password,
                    "role": "employee",
                    "employee_code": employee_code,
                    "full_name": full_name,
                    "department": department,
                },
                timeout=30,
            )

            if response.ok:
                st.success("Registration successful. Please login.")
            else:
                st.error(response.text)

    st.stop()


user = st.session_state.user or {}
role = user.get("role", "")

st.sidebar.success(
    f"Logged in: {user.get('username', 'user')} ({role})"
)

if st.sidebar.button("Logout"):
    st.session_state.token = None
    st.session_state.user = None
    st.rerun()

tab_names = ["Dashboard", "My Leave", "AI Assistant"]

if role in {"manager", "admin"}:
    tab_names.append("Approvals")

if role == "admin":
    tab_names.extend(["Leave Types", "Balances", "Users"])

pages = st.tabs(tab_names)


with pages[0]:
    st.header("Dashboard")
    response = api("GET", "/dashboard/me")

    if response and response.ok:
        st.json(response.json())
    elif response:
        st.error(response.text)


with pages[1]:
    st.header("My Leave")

    response = api("GET", "/leave-balances/me")
    if response and response.ok:
        st.subheader("Balances")
        st.dataframe(response.json(), use_container_width=True)

    st.subheader("Apply for Leave")

    response = api("GET", "/leave-types")
    type_options = {}

    if response and response.ok:
        type_options = {
            item["name"]: item["id"]
            for item in response.json()
            if item["is_active"]
        }

    if type_options:
        selected = st.selectbox(
            "Leave type",
            list(type_options.keys()),
        )
        start = st.date_input("Start date")
        end = st.date_input("End date")
        reason = st.text_area("Reason")

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
                st.success("Leave request submitted.")
            elif response:
                st.error(response.text)
    else:
        st.info("No active leave types configured.")

    response = api("GET", "/leave-requests/me")
    if response and response.ok:
        st.subheader("My Requests")
        st.dataframe(response.json(), use_container_width=True)


with pages[2]:
    st.header("🤖 Policy AI Assistant")

    question = st.text_area(
        "Ask a question about the company leave policy",
        placeholder="What information is required to apply for leave?",
    )

    if st.button("Ask AI", type="primary"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            response = api(
                "POST",
                "/ai/query",
                json={
                    "question": question,
                    "top_k": 3,
                },
            )

            if response and response.ok:
                data = response.json()

                st.markdown("### Answer")
                st.write(data["answer"])

                st.caption(
                    f"Grounded: {data['grounded']} | "
                    f"Hallucination score: {data['hallucination_score']}"
                )

                st.markdown("### Sources")
                st.json(data["sources"])
            elif response:
                st.error(response.text)


if role in {"manager", "admin"}:
    with pages[3]:
        st.header("Leave Approvals")

        response = api("GET", "/leave-requests/pending")

        if response and response.ok:
            requests_data = response.json()

            if not requests_data:
                st.info("No pending requests.")

            for item in requests_data:
                with st.expander(
                    f"Request #{item['id']} — "
                    f"{item['start_date']} to {item['end_date']}"
                ):
                    st.write(f"Employee ID: {item['employee_id']}")
                    st.write(f"Days: {item['days']}")
                    st.write(f"Reason: {item['reason']}")

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
                                json={"comment": comment},
                            )

                            if rr and rr.ok:
                                st.success("Approved.")
                                st.rerun()
                            elif rr:
                                st.error(rr.text)

                    with col2:
                        if st.button(
                            "Reject",
                            key=f"reject_{item['id']}",
                        ):
                            rr = api(
                                "POST",
                                f"/leave-requests/{item['id']}/reject",
                                json={"comment": comment},
                            )

                            if rr and rr.ok:
                                st.success("Rejected.")
                                st.rerun()
                            elif rr:
                                st.error(rr.text)


if role == "admin":
    with pages[4]:
        st.header("Leave Types")

        response = api("GET", "/leave-types")
        if response and response.ok:
            st.dataframe(
                response.json(),
                use_container_width=True,
            )

        name = st.text_input("New leave type name")
        description = st.text_area("Description")

        if st.button("Create Leave Type"):
            response = api(
                "POST",
                "/leave-types",
                json={
                    "name": name,
                    "description": description,
                },
            )

            if response and response.ok:
                st.success("Created.")
                st.rerun()
            elif response:
                st.error(response.text)

    with pages[5]:
        st.header("Balances")
        st.info(
            "Use Swagger to create and update employee leave balances."
        )

    with pages[6]:
        st.header("Users")

        response = api("GET", "/users")
        if response and response.ok:
            st.dataframe(
                response.json(),
                use_container_width=True,
            )
