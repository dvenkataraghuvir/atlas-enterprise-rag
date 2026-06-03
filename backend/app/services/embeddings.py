from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    BAAI/bge-base-en-v1.5 — 768-dimensional embeddings.
    Runs fully locally, no API key, no rate limits.
    Top-ranked on MTEB retrieval leaderboard for its size class.
    Downloads ~440 MB on first run, cached permanently after.
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            # BGE models use a query prefix for retrieval tasks
            "batch_size": 32,
        },
    )
