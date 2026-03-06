"""FastAPI application with RAG-powered search and query endpoints."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from muenster4you.rag.config import config
from muenster4you.rag.generation import RAGGenerator
from muenster4you.rag.retrieval import WikiRetriever

# Lazy-initialized singletons
_retriever: WikiRetriever | None = None
_generator: RAGGenerator | None = None


def get_retriever() -> WikiRetriever:
    global _retriever
    if _retriever is None:
        _retriever = WikiRetriever()
    return _retriever


def get_generator() -> RAGGenerator:
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up on startup
    get_retriever()
    get_generator()
    yield


app = FastAPI(title="Muenster4You API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int = config.default_top_k
    temperature: float = config.default_temperature


class SearchResultItem(BaseModel):
    page_id: int
    page_title: str
    content_text: str
    similarity_score: float
    page_len: int
    source: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchResultItem]


@app.get("/")
async def root():
    return {
        "service": "muenster4you",
        "version": "0.1.0",
        "status": "ok",
    }


@app.get("/search")
async def search(
    q: Optional[str] = Query(None, description="Search query string"),
    top_k: int = Query(config.default_top_k, ge=1, le=20),
):
    if not q:
        return {"message": "Please provide a search query", "query": None}

    retriever = get_retriever()
    results = retriever.retrieve(q, top_k=top_k)

    return {
        "query": q,
        "results": [
            SearchResultItem(
                page_id=r.page_id,
                page_title=r.page_title,
                content_text=r.content_text,
                similarity_score=r.similarity_score,
                page_len=r.page_len,
                source=r.source,
            )
            for r in results
        ],
    }


@app.post("/query")
async def query(req: QueryRequest) -> QueryResponse:
    retriever = get_retriever()
    generator = get_generator()

    results = retriever.retrieve(req.question, top_k=req.top_k)
    answer = generator.generate(req.question, results, temperature=req.temperature)

    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=[
            SearchResultItem(
                page_id=r.page_id,
                page_title=r.page_title,
                content_text=r.content_text,
                similarity_score=r.similarity_score,
                page_len=r.page_len,
                source=r.source,
            )
            for r in results
        ],
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
