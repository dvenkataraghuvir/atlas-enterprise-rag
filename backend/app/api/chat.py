from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    # TODO: wire to RAG pipeline
    async def generate():
        yield 'data: {"token": "Atlas backend connected."}\n\n'
        yield 'data: {"done": true}\n\n'
    return StreamingResponse(generate(), media_type="text/event-stream")
