from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, ingest as ingest_router
from app.services.ingest import ensure_collection_exists, collection_is_empty, ingest_sample_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: ensure Qdrant collection exists and auto-ingest sample docs
    try:
        ensure_collection_exists()
        if collection_is_empty():
            print("Collection is empty — ingesting sample documents...")
            results = ingest_sample_docs()
            print(f"Auto-ingested: {results}")
    except Exception as e:
        print(f"Startup ingest warning (Qdrant may not be ready yet): {e}")
    yield


app = FastAPI(
    title="Atlas — Enterprise RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,           prefix="/api", tags=["chat"])
app.include_router(ingest_router.router,  prefix="/api", tags=["ingest"])


@app.get("/health")
def health():
    return {"status": "ok", "project": "atlas"}
