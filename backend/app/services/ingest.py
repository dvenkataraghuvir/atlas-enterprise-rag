import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.services.embeddings import get_embeddings
from app.core.config import get_settings


def _get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection_exists():
    """Create the Qdrant collection if it doesn't already exist."""
    settings = get_settings()
    client = _get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def collection_is_empty() -> bool:
    settings = get_settings()
    client = _get_qdrant_client()
    try:
        count = client.count(collection_name=settings.qdrant_collection).count
        return count == 0
    except Exception:
        return True


def ingest_file(file_path: str) -> dict:
    """Ingest a single .txt or .pdf file into Qdrant."""
    settings = get_settings()
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        loader = PyMuPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    Qdrant.from_documents(
        chunks,
        embeddings,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    return {"file": path.name, "chunks_ingested": len(chunks)}


def ingest_sample_docs():
    """Auto-ingest the bundled sample documents on first startup."""
    sample_dir = Path(__file__).parent.parent.parent.parent / "docs" / "sample-docs"
    if not sample_dir.exists():
        return
    results = []
    for f in sample_dir.iterdir():
        if f.suffix.lower() in {".txt", ".pdf", ".md"}:
            results.append(ingest_file(str(f)))
    return results
