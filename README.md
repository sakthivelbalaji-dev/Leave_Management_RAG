# 🤖 Leave Management AI

> An AI-powered Leave Management System that combines Natural Language Processing, Retrieval-Augmented Generation (RAG), database-backed leave operations, role-based access control, and a Streamlit interface.

---

## 📌 Overview

Leave Management AI is an intelligent leave-management platform that allows employees and managers to interact with the system using natural language.

### Employees can

- Ask company-policy questions
- Check leave balance
- View leave history
- Check leave status
- Apply for leave using natural language
- Cancel leave requests

### Managers can

- View today's leave requests
- View pending leave requests
- View today's and pending requests
- Approve leave requests
- Reject leave requests

The system uses **RAG for company-policy questions** and the **database as the source of truth for employee and leave information**.

---

## ✨ Features

### 👨‍💼 Employee

- Natural-language leave application
- One-day leave detection
- Multi-day leave requests
- Leave balance lookup
- Leave history
- Leave status
- Leave cancellation
- Company-policy Q&A

### 👨‍💼 Manager

- Today's leave requests
- Pending leave requests
- Today's + pending requests
- Approve leave requests
- Reject leave requests

### 🤖 AI

- LLM-based intent understanding
- Natural-language information extraction
- RAG-based policy retrieval
- Qwen3 embeddings
- FAISS vector similarity search
- Grounded responses
- Hallucination checking
- Temporary conversation context
- Confirmation workflow for leave submission

### 🔐 Security

- JWT authentication
- Password hashing
- Role-based authorization
- Employee/manager access separation
- Backend-side permission validation
- Environment-variable secrets

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │      Streamlit       │
                         │     Frontend UI      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │      REST API        │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       Authentication        Leave Management          AI Layer
              │                     │                     │
              │                     ▼                     ▼
              │                 PostgreSQL        Intent Classification
              │                                           │
              │                              ┌────────────┴────────────┐
              │                              │                         │
              │                              ▼                         ▼
              │                         Database                    RAG
              │                                                          │
              │                                                          ▼
              │                                                   Qwen3 Embeddings
              │                                                          │
              │                                                          ▼
              │                                                        FAISS
              │                                                          │
              │                                                          ▼
              │                                                   Policy Context
              │                                                          │
              │                                                          ▼
              │                                                       Groq LLM
              │                                                          │
              │                                                          ▼
              │                                               Hallucination Check
              │
              └──────────────────────────────────────────────────────────────
```

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Authentication | JWT |
| Validation | Pydantic |
| LLM | Groq |
| LLM Model | `openai/gpt-oss-20b` |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` |
| Vector Search | FAISS |
| RAG | Custom RAG Pipeline |
| Language | Python |
| Configuration | python-dotenv |

---

# 📂 Project Structure

```text
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
├── .env.example
├── .gitignore
└── README.md
```

---

# 🧠 AI & RAG Architecture

## Intent Classification

The system uses an LLM to understand the user's intent from natural language.

Supported intents include:

```text
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
```

The classifier is designed to understand the **meaning** of a user's message rather than requiring the exact sentence to exist in a predefined example list.

For example:

```text
I have a medical appointment on September 17 and won't be available.
```

can be interpreted as a leave request.

---

# 🔎 RAG Pipeline

Policy questions follow this pipeline:

```text
User Question
      │
      ▼
Intent Classification
      │
      ▼
leave_policy
      │
      ▼
Query Embedding
      │
      ▼
Qwen3-Embedding-0.6B
      │
      ▼
FAISS Similarity Search
      │
      ▼
Relevant Policy Chunks
      │
      ▼
Context Builder
      │
      ▼
Grounded Prompt
      │
      ▼
Groq LLM
      │
      ▼
Hallucination Check
      │
      ▼
Final Answer
```

The company policy document is treated as the source of truth for policy questions.

If the required information is not present in the retrieved policy context, the assistant should refuse to invent an answer.

---

# 🛡️ Hallucination Prevention

The system uses multiple safeguards.

### 1. RAG Retrieval

Only relevant policy chunks are supplied to the LLM.

### 2. Grounded Prompt

The LLM is instructed to answer only from the supplied policy context.

### 3. No Unsupported Company Rules

The system must not invent:

- Leave entitlements
- Leave limits
- Eligibility
- Working hours
- Approval rules
- Attendance rules
- Company policies

### 4. Database Source of Truth

Employee-specific information comes from the database.

```text
Leave Balance
     ↓
PostgreSQL
```

not from the LLM.

### 5. Hallucination Checker

Generated answers are checked for grounding before being returned.

---

# 🗃️ Database Operations

The database is the source of truth for transactional operations.

Examples:

```text
Leave Balance
Leave Requests
Leave Status
Employee Information
Manager Relationships
Leave Types
```

The AI determines what the user is trying to do.

The backend performs the actual operation.

---

# 📝 Leave Application

Users can apply for leave naturally.

## One-Day Leave

```text
I need sick leave on 17-09-2026 because I have a medical checkup.
```

The system extracts:

```text
Leave Type : Sick Leave
Start Date : 2026-09-17
End Date   : 2026-09-17
Reason     : Medical checkup
```

The end date is automatically set to the start date for a clear single-day request.

## Multi-Day Leave

```text
I need casual leave from 17-09-2026 to 19-09-2026 for personal work.
```

The system extracts:

```text
Leave Type : Casual Leave
Start Date : 2026-09-17
End Date   : 2026-09-19
Reason     : Personal work
```

---

# ✅ Confirmation Workflow

Leave submission uses a confirmation step.

```text
User
 │
 ▼
Leave Request
 │
 ▼
Extract Details
 │
 ▼
Validate
 │
 ▼
Create Draft
 │
 ▼
Ask Confirmation
 │
 ├── Yes ──► Submit to Database
 │
 └── No ───► Cancel
```

Example:

```text
User:
I need sick leave on 17-09-2026 for a medical checkup.

Assistant:
I have prepared your leave request.
Would you like me to submit it?

User:
Yes, submit it.

Assistant:
Leave request submitted successfully.
```

---

# 👨‍💼 Manager Workflow

Managers can interact using natural language.

### Today's Leave Requests

```text
Show today's leave requests.
```

### Pending Requests

```text
Show pending leave requests.
```

### Combined Query

```text
Show today's and pending leave requests.
```

### Approve

```text
Approve leave request 15.
```

### Reject

```text
Reject leave request 15.
```

Authorization is enforced by the backend.

The LLM cannot bypass manager permissions.

---

# 🔐 Authentication

The application uses JWT authentication.

```text
Username + Password
        │
        ▼
Authentication
        │
        ▼
JWT Access Token
        │
        ▼
Authorization: Bearer <token>
        │
        ▼
Authenticated User
        │
        ▼
Role Validation
```

Supported roles:

```text
employee
manager
admin
```

---

# 🔌 API Endpoints

Base URL:

```text
http://127.0.0.1:8000
```

## Authentication

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Register user |
| POST | `/auth/login` | Login |
| GET | `/auth/me` | Current authenticated user |

## AI

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ai/query` | Natural-language AI query |

Example:

```json
{
  "question": "What are the working hours?",
  "top_k": 5
}
```

Leave request example:

```json
{
  "question": "I need sick leave on 17-09-2026 for a medical checkup",
  "top_k": 5
}
```

## Leave Requests

| Method | Endpoint | Description |
|---|---|---|
| POST | `/leave-requests` | Create leave request |
| GET | `/leave-requests/me` | Current user's leaves |
| GET | `/leave-requests/pending` | Pending requests |
| GET | `/leave-requests/{id}` | Get leave request |
| POST | `/leave-requests/{id}/approve` | Approve request |
| POST | `/leave-requests/{id}/reject` | Reject request |

## Leave Balances

| Method | Endpoint | Description |
|---|---|---|
| GET | `/leave-balances/me` | Current user's balances |
| GET | `/leave-balances/employee/{id}` | Employee balance |
| POST | `/leave-balances` | Create balance |
| PUT | `/leave-balances/{id}` | Update balance |

## Leave Types

| Method | Endpoint | Description |
|---|---|---|
| GET | `/leave-types` | List leave types |
| GET | `/leave-types/{id}` | Get leave type |
| POST | `/leave-types` | Create leave type |
| PUT | `/leave-types/{id}` | Update leave type |
| DELETE | `/leave-types/{id}` | Deactivate leave type |

## Dashboard

| Method | Endpoint | Description |
|---|---|---|
| GET | `/dashboard/me` | User dashboard |
| GET | `/dashboard/manager` | Manager dashboard |
| GET | `/dashboard/admin` | Admin dashboard |

## Users

| Method | Endpoint | Description |
|---|---|---|
| GET | `/users/me/profile` | Current profile |
| GET | `/users` | List users |
| PATCH | `/users/{id}/role` | Change user role |
| PATCH | `/users/{id}/status` | Change user status |

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd leave_management_ai_full_code
```

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Environment Variables

Create a `.env` file in the project root.

Example:

```env
DATABASE_URL=postgresql://USERNAME:PASSWORD@HOST:5432/DATABASE_NAME

SECRET_KEY=YOUR_SECRET_KEY

ACCESS_TOKEN_EXPIRE_MINUTES=1440

GROQ_API_KEY=YOUR_GROQ_API_KEY

GROQ_BASE_URL=https://api.groq.com

GROQ_MODEL=openai/gpt-oss-20b
```

### ⚠️ Never commit `.env`

Use `.env.example` to share the required configuration structure.

---

# 🗄️ Database Setup

Make sure PostgreSQL is running.

Create your database and configure:

```env
DATABASE_URL=...
```

The application uses SQLAlchemy to connect to PostgreSQL.

---

# ▶️ Running the Backend

From the project root:

```powershell
uvicorn leave_management.app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

# 🖥️ Running Streamlit

Open a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit/app.py
```

Frontend:

```text
http://localhost:8501
```

---

# ▶️ Complete Startup

### Terminal 1 — Backend

```powershell
cd leave_management_ai_full_code
.\.venv\Scripts\Activate.ps1
uvicorn leave_management.app.main:app --reload
```

### Terminal 2 — Frontend

```powershell
cd leave_management_ai_full_code
.\.venv\Scripts\Activate.ps1
streamlit run streamlit/app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 Demo Credentials

For security, **real passwords should never be committed to a public GitHub repository**.

Create demo accounts locally through the registration endpoint or registration UI.

### Example Employee

```text
Username: demo_employee
Password: DemoPassword123!
Role: employee
```

### Example Manager

```text
Username: demo_manager
Password: DemoPassword123!
Role: manager
```

> These are example credentials for creating local demo accounts. They are not hard-coded production credentials.

---

# 🧪 Demo Queries

## Employee

```text
What are the working hours?
```

```text
What is my leave balance?
```

```text
Show my leave history.
```

```text
What is the status of my leave?
```

```text
I need sick leave on 17-09-2026 because I have a medical checkup.
```

```text
Yes, submit this leave request.
```

## Manager

```text
Show today's leave requests.
```

```text
Show pending leave requests.
```

```text
Show today's and pending leave requests.
```

```text
Approve leave request 15.
```

```text
Reject leave request 15.
```

---

# 📖 Example Policy Query

### User

```text
What are the working hours?
```

### Example grounded response

```text
Standard working days are Monday to Friday.
Office hours are 9:00 AM to 6:00 PM.
Lunch break is from 1:00 PM to 2:00 PM.
Expected working time is 8 hours per working day.
```

If information is not available in the policy document, the system should not invent an answer.

---

# 📊 AI API Response

The AI API returns structured information such as:

```json
{
  "query": "What are the working hours?",
  "intent": "leave_policy",
  "answer": "...",
  "hallucination_score": 0.0,
  "grounded": true,
  "sources": [],
  "request_id": null,
  "requires_confirmation": false,
  "draft": {}
}
```

---

# 🔒 Security Notes

Before deploying:

- Replace the development `SECRET_KEY`
- Use a production PostgreSQL database
- Never commit `.env`
- Never expose `GROQ_API_KEY`
- Use HTTPS in production
- Use strong passwords
- Restrict CORS appropriately
- Validate all API input
- Keep authorization checks server-side

---

# 🧹 Git Configuration

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
.venv/
venv/
env/

# Environment files
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
```

---

# 📌 Important Design Principle

The LLM is **not the source of truth**.

```text
                USER
                  │
                  ▼
             LLM / Intent
                  │
          Understand intent
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
     DATABASE              RAG
        │                   │
        │                   ▼
        │             Policy Documents
        │                   │
        ▼                   ▼
 Personal Data        Company Policy
        │                   │
        └─────────┬─────────┘
                  ▼
             Final Answer
```

The system uses:

**Database → personal and transactional truth**

**Policy documents → company-policy truth**

**LLM → language understanding and response generation**

This separation reduces hallucination and prevents the model from inventing database or company-policy information.

---

# 👨‍💻 Author

**Leave Management AI Project**

Built using:

- Python
- FastAPI
- Streamlit
- PostgreSQL
- SQLAlchemy
- Qwen3 Embeddings
- FAISS
- Groq
- RAG
