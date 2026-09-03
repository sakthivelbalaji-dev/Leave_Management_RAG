🤖 Leave Management AI

An AI-powered Leave Management System that combines FastAPI, Streamlit, PostgreSQL/SQLAlchemy, RAG, Qwen3 embeddings, FAISS, and a Groq-hosted LLM.

The application supports:

Natural-language leave applications

One-day and multi-day leave requests

Leave balance queries

Employee leave history

Leave status queries

Manager pending-leave queries

Today's leave queries

Manager approval/rejection

Company-policy questions using RAG

Grounding/hallucination checking

JWT authentication and role-based access control

Temporary conversational context in the chat UI

1. Architecture

                         ┌──────────────────────┐
                         │      Streamlit       │
                         │    Web Interface     │
                         └──────────┬───────────┘
                                    │ HTTP
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │       REST API       │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
            Authentication     Leave Management       AI
              / JWT              / Database         /ai/query
                                    │                 │
                                    │                 ▼
                                    │          ┌───────────────┐
                                    │          │ Intent LLM    │
                                    │          └───────┬───────┘
                                    │                  │
                                    │          ┌───────┴────────┐
                                    │          │                │
                                    │          ▼                ▼
                                    │       Database           RAG
                                    │                           │
                                    │                           ▼
                                    │                    Qwen3 Embeddings
                                    │                           │
                                    │                           ▼
                                    │                         FAISS
                                    │                           │
                                    │                           ▼
                                    │                     Policy Context
                                    │                           │
                                    │                           ▼
                                    │                    Groq LLM Answer
                                    │                           │
                                    │                           ▼
                                    │                  Hallucination Check
                                    ▼                           │
                              PostgreSQL ◄─────────────────────┘

2. Main Technologies

Component

Technology

Backend API

FastAPI

Frontend

Streamlit

Database

PostgreSQL + SQLAlchemy

Authentication

JWT / HTTP Bearer

Password hashing

bcrypt

LLM

Groq

Embedding model

Qwen3-Embedding-0.6B

Vector search

FAISS

RAG

Custom retrieval pipeline

Validation

Pydantic

HTTP client

Requests

Environment configuration

python-dotenv

The project requirements include FastAPI, Uvicorn, SQLAlchemy, psycopg2-binary, python-dotenv, Pydantic, JWT/security packages, Groq, sentence-transformers, FAISS, NumPy, Streamlit and Requests.

3. Project Structure

leave_management_ai_full_code/
│
├── knowledge_base/
│   └── leave_policy.txt
│
├── rag/
│   ├── chunker.py
│   ├── context_builder.py
│   ├── embeddings.py
│   ├── hallucination_checker.py
│   ├── initializer.py
│   ├── intent_classifier.py
│   ├── llm.py
│   ├── loader.py
│   ├── prompt_builder.py
│   ├── retriever.py
│   ├── vector_store.py
│   └── __init__.py
│
├── leave_management/
│   └── app/
│       ├── core/
│       ├── database/
│       ├── models/
│       ├── routers/
│       ├── schemas/
│       ├── services/
│       └── main.py
│
├── streamlit/
│   └── app.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md

4. RAG Components

loader.py

Loads policy documents from the knowledge base.

chunker.py

Splits policy documents into smaller chunks suitable for embedding and retrieval.

embeddings.py

Loads the Qwen3 embedding model and converts text into vector representations.

vector_store.py

Uses FAISS for similarity search.

retriever.py

Retrieves the most relevant policy chunks.

The current retrieval configuration uses a minimum similarity threshold of approximately 0.25.

context_builder.py

Combines retrieved policy information into the context supplied to the LLM.

prompt_builder.py

Enforces the policy grounding rules.

The assistant is instructed to:

Answer only from supplied policy context.

Never invent company rules.

Never guess missing policy information.

Never invent leave balances or entitlements.

Refuse unsupported policy questions.

hallucination_checker.py

Checks whether the generated answer is grounded in the available information.

llm.py

Connects to the Groq API and sends requests to the configured LLM.

initializer.py

Acts as the main AI pipeline/orchestrator.

It connects intent classification, database operations, RAG retrieval, LLM generation, confirmation workflows and response construction.

5. Natural Language Intent System

The AI supports the following application intents:

leave_policy
leave_request
leave_balance
my_leaves
leave_status
today_leaves
pending_leaves
today_and_pending_leaves
approve_leave
reject_leave
cancel_leave
confirm
deny
general

The intended design is that the intent model understands the meaning of a user's message rather than requiring the exact wording to exist in a predefined phrase list.

For example:

"I have a medical appointment on September 17 and won't be available."

can be interpreted as a leave request even when that exact sentence was never hard-coded.

For policy questions, intent classification only determines that the user is asking about policy. The actual answer must come from the retrieved company-policy document.

6. Leave Application Examples

One-day leave

sick leave 17-09-2026 medical checkup

Expected interpretation:

Leave type : Sick Leave
Start date : 2026-09-17
End date   : 2026-09-17
Reason     : medical checkup

The end date should automatically become the same as the start date when the user clearly requests a single-day leave.

Multi-day leave

I need casual leave from 17-09-2026 to 19-09-2026 because of personal work

Expected:

Start date : 2026-09-17
End date   : 2026-09-19

The assistant should ask for missing information only when the user has genuinely not supplied it.

7. Manager AI Operations

Managers can use natural-language requests such as:

Show today's leave requests

Show pending leave requests

Show today's and pending leave requests

Approve leave request 15

Reject leave request 15

Manager authorization is enforced by the backend rather than trusting the LLM.

A manager can only act on requests assigned to that manager.

The backend verifies the manager relationship before approval/rejection.

8. API Endpoints

Base URL:

http://127.0.0.1:8000

Authentication uses:

Authorization: Bearer <access_token>

Authentication

Register

POST /auth/register

Example:

{
  "username": "employee1",
  "email": "employee1@example.com",
  "password": "your-password",
  "role": "employee",
  "employee_code": "EMP001",
  "full_name": "Employee One",
  "department": "Engineering",
  "manager_id": null
}

Supported roles:

employee
manager
admin

Login

POST /auth/login

Example:

{
  "username": "employee1",
  "password": "your-password"
}

Returns an access token.

Current user

GET /auth/me

Requires authentication.

9. AI Endpoint

Natural-language AI query

POST /ai/query

Requires authentication.

Example:

{
  "question": "What are the working hours?",
  "top_k": 5
}

Example leave request:

{
  "question": "I need sick leave on 17-09-2026 for a medical checkup",
  "top_k": 5
}

The authenticated user is passed to the AI pipeline so database operations are performed for the correct user.

The AI endpoint returns information such as:

{
  "query": "...",
  "intent": "leave_request",
  "answer": "...",
  "hallucination_score": 0.0,
  "grounded": true,
  "sources": [],
  "request_id": null,
  "requires_confirmation": true,
  "draft": {}
}

10. Leave Request Endpoints

Submit leave

POST /leave-requests

My leave requests

GET /leave-requests/me

Pending requests

GET /leave-requests/pending

Manager/Admin access.

Get a specific request

GET /leave-requests/{request_id}

Approve

POST /leave-requests/{request_id}/approve

Manager/Admin access.

Optional body:

{
  "comment": "Approved."
}

Reject

POST /leave-requests/{request_id}/reject

Manager/Admin access.

Optional body:

{
  "comment": "Not approved due to project requirements."
}

11. Leave Balance Endpoints

My balances

GET /leave-balances/me

Employee balances

GET /leave-balances/employee/{employee_id}

Manager/Admin access.

Create balance

POST /leave-balances

Admin access.

Update balance

PUT /leave-balances/{balance_id}

Admin access.

12. Leave Type Endpoints

List leave types

GET /leave-types

Get leave type

GET /leave-types/{leave_type_id}

Create leave type

POST /leave-types

Admin access.

Update leave type

PUT /leave-types/{leave_type_id}

Admin access.

Deactivate leave type

DELETE /leave-types/{leave_type_id}

Admin access.

13. Dashboard Endpoints

Current user's dashboard

GET /dashboard/me

The response depends on the authenticated role.

Manager dashboard

GET /dashboard/manager

Manager/Admin access.

Admin dashboard

GET /dashboard/admin

Admin access.

14. User/Admin Endpoints

Current profile

GET /users/me/profile

List users

GET /users

Admin only.

Change user role

PATCH /users/{user_id}/role

Admin only.

Example:

role=manager

Change user status

PATCH /users/{user_id}/status

Admin only.

15. Running the Project

Open PowerShell in the project root:

cd B:\leave_management_ai_full_code

Activate the virtual environment:

.\.venv\Scripts\Activate.ps1

If the virtual environment does not exist:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

16. Environment Configuration

Create:

.env

The application requires database, authentication and Groq configuration.

Typical configuration:

DATABASE_URL=postgresql://USERNAME:PASSWORD@HOST:5432/DATABASE_NAME

SECRET_KEY=CHANGE_THIS_TO_A_LONG_RANDOM_SECRET

ACCESS_TOKEN_EXPIRE_MINUTES=1440

GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_BASE_URL=https://api.groq.com
GROQ_MODEL=openai/gpt-oss-20b

Do not commit .env to GitHub.

Use .env.example for sharing configuration structure without secrets.

17. Start FastAPI Backend

From the project root:

uvicorn leave_management.app.main:app --reload

Backend:

http://127.0.0.1:8000

FastAPI Swagger documentation:

http://127.0.0.1:8000/docs

ReDoc:

http://127.0.0.1:8000/redoc

18. Start Streamlit Frontend

Open a second PowerShell terminal.

Activate the environment:

cd B:\leave_management_ai_full_code
.\.venv\Scripts\Activate.ps1

Run:

streamlit run streamlit/app.py

The Streamlit interface normally opens at:

http://localhost:8501

The frontend uses:

http://127.0.0.1:8000

as the default FastAPI URL.

19. Recommended Startup Order

Terminal 1 — Backend

cd B:\leave_management_ai_full_code
.\.venv\Scripts\Activate.ps1
uvicorn leave_management.app.main:app --reload

Terminal 2 — Frontend

cd B:\leave_management_ai_full_code
.\.venv\Scripts\Activate.ps1
streamlit run streamlit/app.py

Then open:

http://localhost:8501

20. Demo Credentials

The source project does not contain a verified hard-coded demo username/password seed.

Therefore, do not assume credentials such as:

manager / manager123
employee / employee123

unless they have actually been created in the database.

Create a demo employee

Use the Streamlit Register tab or:

POST /auth/register

with:

{
  "username": "demo_employee",
  "email": "demo_employee@example.com",
  "password": "DemoPassword123!",
  "role": "employee",
  "employee_code": "DEMO001",
  "full_name": "Demo Employee",
  "department": "Engineering"
}

Create a demo manager

Create a manager account:

{
  "username": "demo_manager",
  "email": "demo_manager@example.com",
  "password": "DemoPassword123!",
  "role": "manager",
  "employee_code": "MGR001",
  "full_name": "Demo Manager",
  "department": "Engineering"
}

Then associate employees with the manager using the appropriate manager_id.

Important: These are example credentials to create a demo account; they are not credentials that were found pre-seeded in the project database.

21. Demo Flow

Employee demo

Login as an employee.

Ask:

What are the working hours?

Expected policy information:

Monday to Friday
9:00 AM to 6:00 PM
Lunch: 1:00 PM to 2:00 PM
Expected working time: 8 hours

Ask:

What is my leave balance?

The answer comes from the database.

Ask:

About my leave

The answer comes from the employee's database records.

Apply leave:

I need sick leave on 17-09-2026 because I have a medical checkup.

The system should create a draft with:

Leave type: Sick Leave
Start: 2026-09-17
End: 2026-09-17
Reason: medical checkup

The system should request confirmation before submitting the leave.

Then:

Yes, submit this leave request.

The leave is submitted to the database.

22. Manager Demo

Login as a manager.

Ask:

Show me today's leave requests.

Ask:

Show pending leave requests.

Ask:

Show today's and pending leave requests.

Approve:

Approve leave request 15.

Reject:

Reject leave request 15.

The backend verifies manager permissions and assignment before changing a leave request.

23. RAG Demo

Ask:

What are the working hours?

The system:

Question
   ↓
Intent classification
   ↓
leave_policy
   ↓
Qwen embedding
   ↓
FAISS similarity search
   ↓
Relevant policy chunks
   ↓
Context builder
   ↓
Grounded LLM prompt
   ↓
LLM answer
   ↓
Hallucination check

For unsupported information:

What is the company policy for something that isn't documented?

The assistant should refuse instead of inventing a policy.

24. Security Model

The application uses JWT authentication.

After login:

username + password
        ↓
authenticate
        ↓
JWT access token
        ↓
Authorization: Bearer <token>
        ↓
get_current_user()
        ↓
database user
        ↓
role verification

Roles:

employee
manager
admin

Manager/admin permissions are enforced server-side.

The AI model is not trusted with authorization.

25. Hallucination Prevention

The application uses multiple layers of protection.

Layer 1 — Retrieval

Policy answers are based on retrieved documents.

Layer 2 — Prompt restrictions

The prompt explicitly instructs the LLM not to invent company policy.

Layer 3 — Grounding check

The generated response is checked for grounding.

Layer 4 — Database source of truth

Personal leave balances, leave history and leave status come from the database rather than the LLM.

This separation is important:

Company policy
      ↓
     RAG
      ↓
Policy answer

Personal leave data
      ↓
    Database
      ↓
Database answer

User intent
      ↓
Intent model
      ↓
Structured action

26. Important Design Principle

The LLM should understand the user's natural language, but it should not become the source of truth.

For example:

User:
How many sick leaves do I have left?

The model identifies:

leave_balance

But the actual balance must be obtained from the database.

Similarly:

User:
What are the working hours?

The model identifies:

leave_policy

RAG retrieves the policy document and supplies the answer.

The LLM should not invent the working hours.

27. Git / GitHub

Recommended .gitignore:

# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environment
.venv/
venv/
env/

# Environment secrets
.env
.env.*
!.env.example

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Local databases
*.db
*.sqlite
*.sqlite3

# Generated files
*.pyc

Before pushing:

git status
git add .
git commit -m "Complete Leave Management AI"
git push

Never commit:

.env
GROQ_API_KEY
DATABASE_URL containing a password
SECRET_KEY

28. Troubleshooting

Backend does not start

Check:

python --version

Check packages:

pip list

Reinstall:

pip install -r requirements.txt

Database error

Verify:

DATABASE_URL=...

The database configuration is required before SQLAlchemy creates the engine.

Groq error

Verify:

GROQ_API_KEY=...

The application requires a valid Groq API key for LLM operations.

RAG says no documents found

Verify:

knowledge_base/
└── leave_policy.txt

The retriever expects policy documents in the knowledge base.

Streamlit cannot connect

Make sure FastAPI is running first:

uvicorn leave_management.app.main:app --reload

Then run:

streamlit run streamlit/app.py

29. API Quick Reference

Method

Endpoint

Access

POST

/auth/register

Public

POST

/auth/login

Public

GET

/auth/me

Authenticated

POST

/ai/query

Authenticated

GET

/dashboard/me

Authenticated

GET

/dashboard/manager

Manager/Admin

GET

/dashboard/admin

Admin

GET

/users/me/profile

Authenticated

GET

/users

Admin

PATCH

/users/{user_id}/role

Admin

PATCH

/users/{user_id}/status

Admin

GET

/leave-types

Authenticated

GET

/leave-types/{id}

Authenticated

POST

/leave-types

Admin

PUT

/leave-types/{id}

Admin

DELETE

/leave-types/{id}

Admin

GET

/leave-balances/me

Authenticated

GET

/leave-balances/employee/{id}

Manager/Admin

POST

/leave-balances

Admin

PUT

/leave-balances/{id}

Admin

POST

/leave-requests

Employee/Manager/Admin

GET

/leave-requests/me

Authenticated

GET

/leave-requests/pending

Manager/Admin

GET

/leave-requests/{id}

Authenticated

POST

/leave-requests/{id}/approve

Manager/Admin

POST

/leave-requests/{id}/reject

Manager/Admin

30. Quick Start

cd B:\leave_management_ai_full_code

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

Configure .env.

Start backend:

uvicorn leave_management.app.main:app --reload

Open another terminal and start frontend:

cd B:\leave_management_ai_full_code
.\.venv\Scripts\Activate.ps1
streamlit run streamlit/app.py

Open:

http://localhost:8501

API documentation:

http://127.0.0.1:8000/docs

31. Project Goal

The goal of the project is to provide a secure, natural-language Leave Management System where:

Users speak naturally
        ↓
AI understands intent
        ↓
Database handles personal leave operations
        ↓
RAG handles company-policy knowledge
        ↓
LLM generates grounded responses
        ↓
Hallucination checking validates the response
        ↓
FastAPI returns structured results
        ↓
Streamlit provides the user interface

The system therefore combines LLM reasoning, RAG retrieval, vector search, database-backed operations, authentication, authorization and a conversational interface without making the LLM itself the source of truth for company or employee data.