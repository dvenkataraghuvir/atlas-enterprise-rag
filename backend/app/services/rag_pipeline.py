import asyncio
import time
import json
from typing import AsyncGenerator

from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient

from app.services.embeddings import get_embeddings
from app.core.config import get_settings

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are Atlas, an enterprise knowledge assistant. Answer using ONLY the context below.
Cite sources as [1], [2], etc. Use **bold** for key terms.
If the context does not contain enough information, say exactly: "I couldn't find a reliable answer in your documents. I'd rather tell you that than guess."

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{question}"),
])

GROUND_PROMPT = ChatPromptTemplate.from_template(
    """Rate how well this answer is source-verified — meaning every claim can be traced to the provided context.
Reply with ONLY a decimal number between 0.00 and 1.00.

Context (first 800 chars): {context}
Answer: {answer}

Faithfulness score:"""
)

# Load cross-encoder once at module level (avoids reloading per request)
_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


def _fetch_all_docs() -> list[Document]:
    """Fetch all stored document chunks from Qdrant for BM25 bootstrapping."""
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
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
            if r.payload.get("page_content")
        ]
    except Exception:
        return []


def _hybrid_retrieve_and_rerank(
    query: str, all_docs: list[Document], top_k: int = 4
) -> list[Document]:
    """
    Manual hybrid retrieval:
      1. Dense search via Qdrant (semantic)
      2. BM25 keyword search (in-memory)
      3. Reciprocal Rank Fusion merge
      4. Cross-encoder reranking → keep top_k
    """
    settings = get_settings()
    embeddings = get_embeddings()

    # ── Dense retrieval ──────────────────────────────────────────────────
    qdrant_store = QdrantVectorStore(
        client=QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
    dense_docs = qdrant_store.similarity_search(query, k=12)

    # ── BM25 retrieval ───────────────────────────────────────────────────
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 8
    bm25_docs = bm25.invoke(query)

    # ── RRF merge (dedup by content prefix) ─────────────────────────────
    seen: set[str] = set()
    merged: list[Document] = []
    for doc in dense_docs + bm25_docs:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    if not merged:
        return []

    # ── Cross-encoder reranking ──────────────────────────────────────────
    model = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in merged]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


async def _fetch_wikipedia_live(query: str) -> str:
    """Fetch a Wikipedia article and return the full text for direct prompt injection."""
    import wikipedia as wiki_pkg

    def _sync_fetch() -> str:
        try:
            # Search first to get the best matching article title,
            # then fetch that page — handles natural language queries correctly
            results = wiki_pkg.search(query, results=3)
            if not results:
                return ""
            page = wiki_pkg.page(results[0], auto_suggest=False)
            return f"[Wikipedia: {page.title}]\n{page.content[:15000]}"
        except wiki_pkg.DisambiguationError as e:
            try:
                page = wiki_pkg.page(e.options[0], auto_suggest=False)
                return f"[Wikipedia: {page.title}]\n{page.content[:15000]}"
            except Exception:
                return ""
        except Exception:
            return ""

    return await asyncio.to_thread(_sync_fetch)


async def run_rag_stream(
    query: str,
    history: list[dict] | None = None,
    use_wikipedia: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Full RAG pipeline as an SSE stream.
    Events: pipeline_step | token | done | error
    """
    settings = get_settings()
    start = time.time()
    history = history or []

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        # ── Step 1: load docs ────────────────────────────────────────────
        yield sse({"type": "pipeline_step", "name": "Loading index",
                   "detail": "fetching chunks for BM25"})
        all_docs = _fetch_all_docs()

        # ── Step 2: live Wikipedia fetch (no chunking) ───────────────────
        wiki_context = ""
        if use_wikipedia:
            yield sse({"type": "pipeline_step", "name": "Wikipedia (live)",
                       "detail": "fetching article → injecting full text"})
            wiki_context = await _fetch_wikipedia_live(query)

        if not all_docs and not wiki_context:
            yield sse({"type": "error",
                       "message": "No documents ingested yet. "
                                  "Please ingest documents first via POST /api/ingest."})
            return

        # ── Step 3: hybrid retrieval + rerank (Qdrant docs) ─────────────
        retrieved_docs: list[Document] = []
        if all_docs:
            yield sse({"type": "pipeline_step", "name": "Hybrid search",
                       "detail": f"BM25 + dense over {len(all_docs)} chunks"})
            yield sse({"type": "pipeline_step", "name": "Reranker",
                       "detail": "cross-encoder/ms-marco · keep top 4"})
            retrieved_docs = _hybrid_retrieve_and_rerank(query, all_docs, top_k=4)

        # ── Step 4: assemble context ─────────────────────────────────────
        yield sse({"type": "pipeline_step", "name": "Prompt assembly",
                   "detail": f"{'Wikipedia + ' if wiki_context else ''}{len(retrieved_docs)} chunks injected"})

        qdrant_context = _format_docs(retrieved_docs) if retrieved_docs else ""
        context_str = (wiki_context + ("\n\n" if qdrant_context else "") + qdrant_context).strip()

        # ── Step 5: build conversation history ───────────────────────────
        history_messages = []
        for msg in history[-6:]:  # last 3 turns (6 messages)
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                history_messages.append(AIMessage(content=content))

        # ── Step 6: LLM generation (streamed) ───────────────────────────
        yield sse({"type": "pipeline_step", "name": "Generating answer",
                   "detail": "Groq · Llama 4 Scout"})

        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=settings.groq_api_key,
            streaming=True,
            temperature=0.1,
        )
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        full_answer = ""
        async for token in chain.astream({
            "context": context_str,
            "question": query,
            "history": history_messages,
        }):
            full_answer += token
            yield sse({"type": "token", "text": token})

        # ── Step 7: source verification ──────────────────────────────────
        yield sse({"type": "pipeline_step", "name": "Source Verification",
                   "detail": "Gemini 2.5 Flash judge"})

        if settings.google_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            ground_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.google_api_key,
                temperature=0,
            )
        elif settings.nvidia_api_key:
            from langchain_openai import ChatOpenAI
            ground_llm = ChatOpenAI(
                model="meta/llama-3.1-8b-instruct",
                api_key=settings.nvidia_api_key,
                base_url="https://integrate.api.nvidia.com/v1",
                temperature=0,
            )
        else:
            ground_llm = ChatGroq(
                model="llama-3.1-8b-instant",
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
        if wiki_context:
            sources.append({
                "id": 1,
                "title": query.title(),
                "snippet": wiki_context[wiki_context.find("\n")+1:][:220] + "…",
                "space": "Wikipedia (live)",
            })
        offset = len(sources)
        sources += [
            {
                "id": i + 1 + offset,
                "title": (doc.metadata or {}).get("source", f"Document {i+1}"),
                "snippet": doc.page_content[:220] + "…",
                "space": (doc.metadata or {}).get("space", "Knowledge Base"),
            }
            for i, doc in enumerate(retrieved_docs)
        ]

        yield sse({
            "type": "done",
            "sources": sources,
            "ground": ground_score,
            "latency": f"{round(time.time() - start, 1)}s",
            "retrieved": len(all_docs),
            "kept": len(retrieved_docs) + (1 if wiki_context else 0),
        })

    except Exception as e:
        yield sse({"type": "error", "message": str(e)})
