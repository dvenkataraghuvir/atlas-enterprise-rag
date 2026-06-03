from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from app.services.embeddings import get_embeddings
from app.core.config import get_settings


async def ingest_wikipedia(query: str, max_articles: int = 2) -> dict:
    """
    Search Wikipedia for a topic, fetch the articles and ingest into Qdrant.
    No API key needed — Wikipedia is fully public.
    """
    loader = WikipediaLoader(
        query=query,
        load_max_docs=max_articles,
        doc_content_chars_max=40000,  # ~8,000 words per article max
    )
    docs = loader.load()

    if not docs:
        return {"query": query, "articles_ingested": 0, "chunks_ingested": 0, "titles": []}

    # Tag every chunk with Wikipedia as the source type
    for doc in docs:
        doc.metadata["source_type"] = "wikipedia"
        doc.metadata["source"] = doc.metadata.get("title", query)
        doc.metadata["url"] = doc.metadata.get("source", "")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    settings = get_settings()
    embeddings = get_embeddings()
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    titles = list({doc.metadata.get("title", query) for doc in docs})
    return {
        "query": query,
        "articles_ingested": len(docs),
        "chunks_ingested": len(chunks),
        "titles": titles,
        "source_type": "wikipedia",
    }
