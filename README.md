# Atlas — Enterprise RAG Knowledge Assistant

> Project 01 of 3 · AI Engineering Portfolio · [dvenkataraghuvir](https://github.com/dvenkataraghuvir)

Atlas answers questions over internal documents — PDFs, wikis, Confluence pages — with hybrid search, cross-encoder reranking, inline citations, and a groundedness check that refuses to answer rather than hallucinate.

## Architecture

```
User Question
  → Hybrid Search (dense + BM25)  — Qdrant
  → Cross-encoder Reranker        — HuggingFace (local)
  → Prompt Assembly
  → LLM Generation (streaming)   — Groq Llama 3.3 70B
  → Groundedness Check            — faithfulness score
  → Answer + Citations + OTel trace → Lens
```

## Stack

| Layer         | Technology                                      |
|---------------|-------------------------------------------------|
| Frontend      | Vanilla HTML / CSS / JS (interactive prototype) |
| API           | FastAPI + Pydantic v2                           |
| Orchestration | LangChain LCEL                                  |
| Vector DB     | Qdrant (Docker)                                 |
| Embeddings    | sentence-transformers/all-MiniLM-L6-v2 (local)  |
| Reranker      | cross-encoder/ms-marco-MiniLM-L-6-v2 (local)    |
| LLM           | Groq — Llama 3.3 70B (free API)                 |
| Observability | OpenTelemetry → Lens (Project 03)               |

## Quickstart

```bash
cp backend/.env.example backend/.env
# Add your GROQ_API_KEY to backend/.env

docker-compose up --build
# Open frontend/index.html in your browser
```

## Project Structure

```
atlas-enterprise-rag/
├── frontend/               # Interactive prototype UI
│   ├── index.html
│   └── assets/ds.css
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI entrypoint
│   │   ├── api/            # Route handlers
│   │   ├── services/       # RAG pipeline, ingestion
│   │   └── models/         # Pydantic schemas
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Related Projects

- [Relay — Multi-Agent Workflow Automation](https://github.com/dvenkataraghuvir/relay-agentic-workflows)
- [Lens — LLM Eval & Guardrails](https://github.com/dvenkataraghuvir/lens-llm-eval-guardrails)
