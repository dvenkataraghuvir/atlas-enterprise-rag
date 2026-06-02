import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.services.embeddings import get_embeddings
from app.core.config import get_settings

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log",          # plain text
    ".pdf",                                  # PDF
    ".docx", ".doc",                         # Word
    ".pptx", ".ppt",                         # PowerPoint
    ".xlsx", ".xls",                         # Excel
    ".html", ".htm",                         # HTML
    ".csv",                                  # CSV
    # Code files
    ".py",                                   # Python
    ".js", ".jsx", ".ts", ".tsx",            # JavaScript / TypeScript
    ".java",                                 # Java
    ".go",                                   # Go
    ".rs",                                   # Rust
    ".cpp", ".cc", ".c", ".h",              # C / C++
    ".rb",                                   # Ruby
    ".swift",                                # Swift
    ".kt",                                   # Kotlin
    ".cs",                                   # C#
    ".php",                                  # PHP
    ".scala",                                # Scala
    ".sol",                                  # Solidity
    ".sh", ".bash",                          # Shell scripts
}

# Maps file extension → LangChain Language enum for code-aware splitting
_CODE_LANGUAGE_MAP: dict[str, Language] = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".jsx":   Language.JS,
    ".ts":    Language.TS,
    ".tsx":   Language.TS,
    ".java":  Language.JAVA,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".cpp":   Language.CPP,
    ".cc":    Language.CPP,
    ".c":     Language.C,
    ".h":     Language.C,
    ".rb":    Language.RUBY,
    ".swift": Language.SWIFT,
    ".kt":    Language.KOTLIN,
    ".cs":    Language.CSHARP,
    ".php":   Language.PHP,
    ".scala": Language.SCALA,
    ".sol":   Language.SOL,
    ".rst":   Language.RST,
    ".html":  Language.HTML,
    ".htm":   Language.HTML,
}


def _get_loader(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyMuPDFLoader(str(path))
    if ext in {".docx", ".doc"}:
        return Docx2txtLoader(str(path))
    if ext in {".pptx", ".ppt"}:
        return UnstructuredPowerPointLoader(str(path))
    if ext in {".xlsx", ".xls"}:
        return UnstructuredExcelLoader(str(path))
    if ext in {".html", ".htm"}:
        return BSHTMLLoader(str(path), open_encoding="utf-8")
    if ext == ".csv":
        return CSVLoader(str(path))
    # Code files and plain text all load as text
    return TextLoader(str(path), encoding="utf-8")


def _get_splitter(path: Path) -> RecursiveCharacterTextSplitter:
    """Return a language-aware splitter for code, plain splitter for everything else."""
    ext = path.suffix.lower()
    lang = _CODE_LANGUAGE_MAP.get(ext)
    if lang:
        # Splits on function/class boundaries — preserves code structure
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=1000,
            chunk_overlap=100,
        )
    # Documents: split by paragraphs → sentences → words
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection_exists():
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
    """Ingest any supported file into Qdrant with language-aware chunking for code."""
    settings = get_settings()
    path = Path(file_path)

    loader = _get_loader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = path.suffix.lower()

    splitter = _get_splitter(path)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )
    return {"file": path.name, "chunks_ingested": len(chunks), "type": path.suffix.lower()}


def ingest_sample_docs():
    """Auto-ingest bundled sample docs on first startup."""
    sample_dir = Path(__file__).parent.parent.parent.parent / "docs" / "sample-docs"
    if not sample_dir.exists():
        return []
    results = []
    for f in sorted(sample_dir.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                results.append(ingest_file(str(f)))
            except Exception as e:
                results.append({"file": f.name, "error": str(e)})
    return results
