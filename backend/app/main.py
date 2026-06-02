from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, ingest

app = FastAPI(title="Atlas — Enterprise RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,   prefix="/api", tags=["chat"])
app.include_router(ingest.router, prefix="/api", tags=["ingest"])

@app.get("/health")
def health():
    return {"status": "ok", "project": "atlas"}
