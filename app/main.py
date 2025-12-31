from typing import Any, Dict, List, Optional
import os
from openai import OpenAI
import textwrap
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # requires: pip install -U langchain-chroma

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "chroma_db"

app = FastAPI(title="Germany RAG Assistant", version="0.1.0")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def format_context(chunks):
    # Build a compact, citeable context for the LLM.
    parts = []
    for i, c in enumerate(chunks, 1):
        cit = c.citation
        cite = f"[{i}] {cit.authority} | {cit.source_file} | page {cit.page}"
        parts.append(f"{cite}\n{c.text}")
    return "\n\n".join(parts)


# --- Models (API contracts) ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question to search for")
    k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve")
    category: Optional[str] = Field(
        None, description="Optional category filter: student/post-study/work"
    )
    use_mmr: bool = Field(True, description="Use MMR for more diverse retrieval")
    debug: bool = Field(False, description="Include extra debug fields in the response")


class Citation(BaseModel):
    authority: Optional[str] = None
    title: Optional[str] = None
    source_file: Optional[str] = None
    page: Optional[int] = None
    doc_id: Optional[str] = None
    category: Optional[str] = None


class RetrievedChunk(BaseModel):
    text: str
    citation: Citation
    score: Optional[float] = (
        None  # Present when use_mmr=false (similarity_search_with_score)
    )


class QueryResponse(BaseModel):
    question: str
    results: List[RetrievedChunk]


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    used_chunks: List[RetrievedChunk]


# --- Load vector DB once at startup ---
_vectordb: Optional[Chroma] = None


@app.on_event("startup")
def load_vectordb() -> None:
    global _vectordb

    if not os.path.exists(PERSIST_DIR):
        raise RuntimeError(
            f"Missing vector DB directory: {PERSIST_DIR}. "
            "Run: python scripts/ingest.py"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    _vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "db_loaded": _vectordb is not None}


def _build_filter(category: Optional[str]) -> Optional[Dict[str, Any]]:
    if not category:
        return None
    return {"category": category}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if _vectordb is None:
        raise RuntimeError("Vector DB not loaded")

    filt = _build_filter(req.category)

    results: List[RetrievedChunk] = []

    if req.use_mmr:
        docs = _vectordb.max_marginal_relevance_search(
            req.question,
            k=req.k,
            fetch_k=max(20, req.k * 5),
            filter=filt,
        )

        for d in docs:
            meta = d.metadata or {}
            citation = Citation(
                authority=meta.get("authority"),
                title=meta.get("title"),
                source_file=meta.get("source_file"),
                page=meta.get("page"),
                doc_id=meta.get("doc_id"),
                category=meta.get("category"),
            )
            results.append(
                RetrievedChunk(
                    text=d.page_content.strip(), citation=citation, score=None
                )
            )

    else:
        pairs = _vectordb.similarity_search_with_score(
            req.question,
            k=req.k,
            filter=filt,
        )

        for d, score in pairs:
            meta = d.metadata or {}
            citation = Citation(
                authority=meta.get("authority"),
                title=meta.get("title"),
                source_file=meta.get("source_file"),
                page=meta.get("page"),
                doc_id=meta.get("doc_id"),
                category=meta.get("category"),
            )
            results.append(
                RetrievedChunk(
                    text=d.page_content.strip(),
                    citation=citation,
                    score=float(score),
                )
            )

    return QueryResponse(question=req.question, results=results)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: QueryRequest) -> AnswerResponse:
    # 1) Retrieve evidence first (same logic as /query)
    qr = query(req)  # reuse your query() logic
    used = qr.results

    # 2) Build strict prompt to prevent hallucination
    context = format_context(used)
    system = (
        "You are a careful assistant answering questions about German residence/work rules. "
        "You MUST use only the provided sources. If the sources do not contain enough "
        "information, say you don't know and ask what document or detail is missing. "
        "Always cite sources using [1], [2], ... matching the provided context blocks."
    )
    user = f"Question:\n{req.question}\n\nSources:\n{context}\n\nWrite a short, clear answer with citations."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    # 3) Call the LLM
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    answer_text = resp.choices[0].message.content.strip()

    # 4) Return answer + evidence + citations
    citations = [c.citation for c in used]
    return AnswerResponse(
        question=req.question,
        answer=answer_text,
        citations=citations,
        used_chunks=used,
    )
