from typing import Any

from pydantic import BaseModel, Field



class ChatMessage(BaseModel):
    role: str
    content: str




class AIQueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    # Used when confirming a previously prepared action
    confirmed: bool = False

    # Leave draft prepared by the AI
    draft: dict[str, Any] | None = None


    #
    # Streamlit sends the previous messages here.
    #
    # This is NOT stored in the database.
    # It exists only for the current Streamlit session.
    #
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
    )




class AISource(BaseModel):
    source: str
    score: float




class AIQueryResponse(BaseModel):
    query: str

    intent: str

    answer: str

    hallucination_score: float = 0.0

    grounded: bool = True

    sources: list[AISource] = Field(
        default_factory=list,
    )

    # Database / request tracking
    request_id: int | None = None

    # Leave application confirmation flow
    requires_confirmation: bool = False

    draft: dict[str, Any] = Field(
        default_factory=dict,
    )