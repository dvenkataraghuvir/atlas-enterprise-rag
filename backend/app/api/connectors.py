from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.connectors.wikipedia import ingest_wikipedia

router = APIRouter()


class WikipediaRequest(BaseModel):
    query: str
    max_articles: int = 2


@router.post("/connectors/wikipedia")
async def connect_wikipedia(request: WikipediaRequest):
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    if request.max_articles < 1 or request.max_articles > 5:
        raise HTTPException(400, "max_articles must be between 1 and 5")

    result = await ingest_wikipedia(request.query, request.max_articles)

    if result["articles_ingested"] == 0:
        raise HTTPException(404, f"No Wikipedia articles found for: '{request.query}'")

    return result


@router.get("/connectors/sources")
async def list_sources():
    """Return all unique source types currently stored in Qdrant."""
    from qdrant_client import QdrantClient
    from app.core.config import get_settings

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    try:
        records, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        sources: dict[str, set] = {}
        for r in records:
            src_type = r.payload.get("metadata", {}).get("source_type", "upload")
            src_name = r.payload.get("metadata", {}).get("source", "Unknown")
            if src_type not in sources:
                sources[src_type] = set()
            sources[src_type].add(src_name)

        return {
            src_type: sorted(list(names))
            for src_type, names in sources.items()
        }
    except Exception as e:
        return {"error": str(e)}
