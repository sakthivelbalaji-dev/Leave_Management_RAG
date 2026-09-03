from groq import Groq

from leave_management.app.core.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
)


class GroqLLM:
    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing from .env")

        # With the Groq Python SDK, use https://api.groq.com here.
        # The SDK adds the OpenAI-compatible /openai/v1 path.
        self.client = Groq(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )
        self.model = GROQ_MODEL

        print(f"Groq model: {self.model}")
        print(f"Groq base URL: {GROQ_BASE_URL}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=700,
        )

        return response.choices[0].message.content.strip()
