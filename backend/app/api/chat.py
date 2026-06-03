from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest
from app.services.rag_pipeline import run_rag_stream

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        run_rag_stream(
            query=request.query,
            history=[m.model_dump() for m in request.history],
            use_wikipedia=request.use_wikipedia,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
