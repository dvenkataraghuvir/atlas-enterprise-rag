import tempfile, os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingest import ingest_file, ensure_collection_exists, SUPPORTED_EXTENSIONS

router = APIRouter()


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    ensure_collection_exists()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = ingest_file(tmp_path)
        result["original_name"] = file.filename
        return result
    finally:
        os.unlink(tmp_path)


@router.get("/ingest/status")
async def ingest_status():
    from qdrant_client import QdrantClient
    from app.core.config import get_settings
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    try:
        count = client.count(collection_name=settings.qdrant_collection).count
        return {"collection": settings.qdrant_collection, "total_chunks": count}
    except Exception as e:
        return {"error": str(e)}
