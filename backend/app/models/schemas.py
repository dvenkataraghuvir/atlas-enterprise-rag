from pydantic import BaseModel


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    collection: str = "documents"
    history: list[HistoryMessage] = []
    use_wikipedia: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    faithfulness: float | None = None
