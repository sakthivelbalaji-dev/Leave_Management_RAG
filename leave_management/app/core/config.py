import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME_IN_ENV")
ALGORITHM = os.getenv("ALGORITHM", os.getenv("ALGORITHUM", "HS256"))
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
