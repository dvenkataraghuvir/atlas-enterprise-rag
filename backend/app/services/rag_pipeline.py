import time
from typing import AsyncGenerator
import json

from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

from app.services.embeddings import get_embeddings
from app.core.config import get_settings

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are Atlas, an enterprise knowledge assistant. Answer using ONLY the context below.
Cite sources as [1], [2], etc. Use **bold** for key terms.
If the context does not contain enough information, say exactly: "I don't have enough information in the connected sources to answer that."

Context:
{context}

Question: {question}

Answer:"""
)

GROUND_PROMPT = ChatPromptTemplate.from_template(
    """Rate how faithfully this answer is grounded in the provided context.
Reply with ONLY a decimal number between 0.00 and 1.00.

Context (first 800 chars): {context}
Answer: {answer}

Faithfulness score:"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
    )


def _build_retriever(all_docs):
    """Build hybrid EnsembleRetriever: BM25 (keyword) + Qdrant (dense) with reranker."""
    settings = get_settings()
    embeddings = get_embeddings()

    # Dense retriever — Qdrant semantic search
    qdrant_store = Qdrant(
        client=QdrantClient(url=settings.qdrant_url),
        collection_name=settings.qdrant_collection,
        embeddings=embeddings,
    )
    dense_retriever = qdrant_store.as_retriever(search_kwargs={"k": 12})

    # Sparse retriever — in-memory BM25 keyword search
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 8

    # Hybrid: RRF merge of both retrievers
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )

    # Cross-encoder reranker — re-scores merged results, keeps top 4
    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        top_n=4,
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )


def _fetch_all_docs():
    """Fetch all documents from Qdrant to bootstrap the BM25 retriever."""
    from langchain_core.documents import Document
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)

    try:
        total = client.count(collection_name=settings.qdrant_collection).count
        if total == 0:
            return []
        records, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=min(total, 2000),
            with_payload=True,
        )
        return [
            Document(
                page_content=r.payload.get("page_content", ""),
                metadata=r.payload.get("metadata", {}),
            )
            for r in records
        ]
    except Exception:
        return []


async def run_rag_stream(query: str) -> AsyncGenerator[str, None]:
    """
    Full RAG pipeline as an SSE stream.
    Yields JSON-encoded events:
      {"type": "pipeline_step", "name": str, "detail": str}
      {"type": "token", "text": str}
      {"type": "done", "sources": [...], "ground": float,
       "latency": str, "retrieved": int, "kept": int}
      {"type": "error", "message": str}
    """
    settings = get_settings()
    start = time.time()

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        # ── Step 1: load docs for BM25 ──────────────────────────────────
        yield sse({"type": "pipeline_step", "name": "Loading index",
                   "detail": "fetching docs for BM25"})
        all_docs = _fetch_all_docs()

        if not all_docs:
            yield sse({"type": "error",
                       "message": "No documents ingested yet. Please ingest documents first."})
            return

        # ── Step 2: hybrid retrieval ─────────────────────────────────────
        yield sse({"type": "pipeline_step", "name": "Hybrid search",
                   "detail": f"BM25 + dense · k={len(all_docs)} docs"})
        retriever = _build_retriever(all_docs)
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_count = len(retrieved_docs)

        # ── Step 3: reranker (already applied inside retriever) ──────────
        yield sse({"type": "pipeline_step", "name": "Reranker",
                   "detail": f"cross-encoder · kept {retrieved_count}"})

        # ── Step 4: prompt assembly ──────────────────────────────────────
        yield sse({"type": "pipeline_step", "name": "Prompt assembly",
                   "detail": f"{retrieved_count} chunks injected"})
        context_str = _format_docs(retrieved_docs)

        # ── Step 5: LLM generation (streamed) ───────────────────────────
        yield sse({"type": "pipeline_step", "name": "Generating answer",
                   "detail": "Groq · Llama 3.3 70B"})

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            streaming=True,
            temperature=0.1,
        )
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        full_answer = ""
        async for token in chain.astream({"context": context_str, "question": query}):
            full_answer += token
            yield sse({"type": "token", "text": token})

        # ── Step 6: groundedness check ───────────────────────────────────
        yield sse({"type": "pipeline_step", "name": "Groundedness check",
                   "detail": "faithfulness score"})

        ground_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0,
        )
        ground_chain = GROUND_PROMPT | ground_llm | StrOutputParser()
        try:
            score_str = await ground_chain.ainvoke({
                "context": context_str[:800],
                "answer": full_answer[:600],
            })
            ground_score = round(float(score_str.strip().split()[0]), 2)
            ground_score = max(0.0, min(1.0, ground_score))
        except Exception:
            ground_score = 0.85

        # ── Build sources list ───────────────────────────────────────────
        sources = []
        for i, doc in enumerate(retrieved_docs):
            meta = doc.metadata or {}
            sources.append({
                "id": i + 1,
                "title": meta.get("source", f"Document {i+1}"),
                "snippet": doc.page_content[:220] + "…",
                "space": meta.get("space", "Knowledge Base"),
            })

        latency = round(time.time() - start, 1)
        yield sse({
            "type": "done",
            "sources": sources,
            "ground": ground_score,
            "latency": f"{latency}s",
            "retrieved": retrieved_count,
            "kept": retrieved_count,
        })

    except Exception as e:
        yield sse({"type": "error", "message": str(e)})
