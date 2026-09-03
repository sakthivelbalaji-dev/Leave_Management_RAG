# Leave Management AI

FastAPI + PostgreSQL + JWT + Groq + RAG/FAISS + Streamlit.

## Important Groq configuration

Use:

GROQ_BASE_URL=https://api.groq.com
GROQ_MODEL=openai/gpt-oss-20b

Do not set GROQ_BASE_URL to https://api.groq.com/openai/v1 when using the Groq Python SDK in this project. That causes the duplicated path:

/openai/v1/openai/v1/chat/completions

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in PostgreSQL and Groq values.

## Run FastAPI

```powershell
uvicorn leave_management.app.main:app --reload
```

Swagger:
http://127.0.0.1:8000/docs

## Run Streamlit

In another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit/app.py
```

## Demo order

1. Register an admin.
2. Register a manager.
3. Register an employee with manager_id set to the manager's user ID.
4. Login and authorize Swagger.
5. Admin creates leave types.
6. Admin allocates leave balances.
7. Employee submits a leave request.
8. Manager approves or rejects it.
9. Employee checks balance/history.
10. Test the AI endpoint.

The included policy file is TEST DATA. Replace it with the approved company policy before final presentation.
