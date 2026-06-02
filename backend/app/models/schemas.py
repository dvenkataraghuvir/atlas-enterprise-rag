from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    collection: str = "documents"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    faithfulness: float | None = None
