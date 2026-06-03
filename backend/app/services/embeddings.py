from functools import lru_cache
from langchain_core.embeddings import Embeddings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """
    Returns Gemini embedding model if GOOGLE_API_KEY is set,
    falls back to local HuggingFace model for offline/dev use.
    Gemini: zero RAM cost on server — model runs on Google's infrastructure.
    HuggingFace: ~200MB RAM, runs fully offline.
    """
    from app.core.config import get_settings
    settings = get_settings()

    if settings.google_api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.google_api_key,
        )

    # Fallback — local HuggingFace model (no API key needed)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
