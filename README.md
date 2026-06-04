# Atlas — Enterprise RAG Knowledge Assistant

> Project 01 of 3 · AI Engineering Portfolio · [dvenkataraghuvir](https://github.com/dvenkataraghuvir)

Atlas answers questions over internal documents — PDFs, wikis, code files — with hybrid search, cross-encoder reranking, inline citations, conversation memory, and a source verification judge that refuses to hallucinate.

## Pipeline

```
User Question
  → Query Rewriting (follow-ups)  — NVIDIA NIM llama-3.2-3b
  → Hybrid Search (BM25 + dense)  — Qdrant Cloud
  → Cross-encoder Reranker        — ms-marco-MiniLM (local)
  → Live Wikipedia injection      — full article, no chunking
  → Conversation Memory           — last 3 turns in context
  → LLM Generation (streaming)   — Groq Llama 4 Scout (500K tokens/day)
  → Source Verification           — Gemini 2.5 Flash judge
  → Answer + Citations
```

## Stack

| Layer | Technology |
|---|---|
| Frontend | React (CDN) · single HTML file, no build step |
| API | FastAPI + Pydantic v2 · SSE streaming |
| Orchestration | LangChain LCEL |
| Vector DB | Qdrant Cloud (free tier · eu-central-1) |
| Embeddings | BAAI/bge-base-en-v1.5 · 768-dim · local, unlimited |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 · local |
| Main LLM | Groq — Llama 4 Scout 17B-16E · 500K tokens/day |
| Query rewriter | NVIDIA NIM — Llama 3.2 3B · no daily limit |
| Source verification | Google Gemini 2.5 Flash · 1,500 req/day |
| Judge fallback | NVIDIA NIM — Llama 3.1 8B · no daily limit |
| Wikipedia | Live fetch per query · full article injected into context |

## Features

- **Hybrid retrieval** — BM25 keyword + dense vector search, merged with RRF, reranked with CrossEncoder
- **Live Wikipedia** — toggle ON to auto-fetch any Wikipedia article per query (no pre-ingestion needed)
- **Conversation memory** — last 3 turns sent with every query; Atlas understands follow-up questions
- **Query rewriting** — follow-up questions like "what is his age?" are rewritten to self-contained queries before retrieval
- **Source Verified badge** — Gemini 2.5 Flash judges every answer for faithfulness; refuses to answer rather than hallucinate
- **SSE streaming** — answer tokens stream in real time with live pipeline trace in the UI
- **35 file types** — PDF, DOCX, PPTX, XLSX, 16 code formats, TXT, HTML, and more

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/dvenkataraghuvir/atlas-enterprise-rag
cd atlas-enterprise-rag/backend
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Fill in: GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
# Optional: GOOGLE_API_KEY (Gemini judge), NVIDIA_API_KEY (NIM fallback)

# 3. Start backend
uvicorn app.main:app --reload --port 8000

# 4. Open frontend
# Open frontend/index.html in your browser
```

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Main LLM (Llama 4 Scout) |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `GOOGLE_API_KEY` | Recommended | Gemini 2.5 Flash source verification judge |
| `NVIDIA_API_KEY` | Recommended | Query rewriting + judge fallback (no daily limit) |

## Project Structure

```
atlas-enterprise-rag/
├── frontend/
│   ├── index.html          # Full React UI — chat thread, Wikipedia toggle, retrieval trace
│   └── assets/ds.css
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI entrypoint + auto-ingestion
│   │   ├── api/            # chat, ingest, connectors routes
│   │   ├── services/
│   │   │   ├── rag_pipeline.py   # Full RAG pipeline with streaming
│   │   │   ├── ingest.py         # Document parsing + chunking
│   │   │   ├── embeddings.py     # BAAI/bge local embeddings
│   │   │   └── connectors/
│   │   │       └── wikipedia.py  # Wikipedia pre-ingestion connector
│   │   └── models/schemas.py
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

## Related Projects

- [Relay — Multi-Agent Workflow Automation](https://github.com/dvenkataraghuvir/relay-multi-agent)
- [Lens — LLM Eval & Guardrails](https://github.com/dvenkataraghuvir/lens-llm-eval)
