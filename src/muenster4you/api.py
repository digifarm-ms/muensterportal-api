"""FastAPI application with RAG-powered search and query endpoints."""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from muenster4you.rag.generation import RAGGenerator
from muenster4you.rag.retrieval import RetrievalResult, WikiRetriever
from muenster4you.rag.sessions import ChatSessionManager


# --- Dependencies ---


@lru_cache
def get_retriever() -> WikiRetriever:
    return WikiRetriever()


@lru_cache
def get_generator() -> RAGGenerator:
    return RAGGenerator()


@lru_cache
def get_session_manager() -> ChatSessionManager:
    return ChatSessionManager()


RetrieverDep = Annotated[WikiRetriever, Depends(get_retriever)]
GeneratorDep = Annotated[RAGGenerator, Depends(get_generator)]
SessionManagerDep = Annotated[ChatSessionManager, Depends(get_session_manager)]


# --- App ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_retriever()
    get_generator()
    get_session_manager()
    yield


app = FastAPI(title="Muenster4You API", version="0.1.0", lifespan=lifespan)


# --- Models ---


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class SearchResultItem(BaseModel):
    page_id: int
    page_title: str
    content_text: str
    similarity_score: float
    page_len: int
    source: str

    @classmethod
    def from_retrieval_result(cls, r: RetrievalResult) -> "SearchResultItem":
        return cls(
            page_id=r.page_id,
            page_title=r.page_title,
            content_text=r.content_text,
            similarity_score=r.similarity_score,
            page_len=r.page_len,
            source=r.source,
        )


class SearchResponse(BaseModel):
    query: str | None
    results: list[SearchResultItem] = []
    message: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchResultItem]


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SearchResultItem]
    history: list[ChatMessage]
    remaining_followups: int


# --- Endpoints ---


@app.get("/")
async def root():
    return {
        "service": "muenster4you",
        "version": "0.1.0",
        "status": "ok",
    }


@app.get("/search")
async def search(
    retriever: RetrieverDep,
    q: Optional[str] = Query(None, description="Search query string"),
    top_k: int = Query(5, ge=1, le=20),
) -> SearchResponse:
    if not q:
        return SearchResponse(query=None, message="Please provide a search query")

    results = retriever.retrieve(q, top_k=top_k)

    return SearchResponse(
        query=q,
        results=[SearchResultItem.from_retrieval_result(r) for r in results],
    )


@app.post("/query")
async def query(
    retriever: RetrieverDep,
    generator: GeneratorDep,
    req: QueryRequest,
) -> QueryResponse:
    results = retriever.retrieve(req.question, top_k=req.top_k)
    answer = generator.generate(req.question, results, temperature=req.temperature)

    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=[SearchResultItem.from_retrieval_result(r) for r in results],
    )


@app.post("/chat")
async def chat(
    retriever: RetrieverDep,
    generator: GeneratorDep,
    session_manager: SessionManagerDep,
    req: ChatRequest,
) -> ChatResponse:
    session_manager.cleanup_expired()

    is_new = True
    session = None
    if req.conversation_id:
        session = session_manager.get_session(req.conversation_id)
        if session is not None:
            is_new = False

    if is_new:
        # First turn: run RAG retrieval and create session
        results = retriever.retrieve(req.message, top_k=req.top_k)
        conversation_id = session_manager.create_session(sources=results)
        system_msg = generator.build_system_message(results)
        session_manager.set_system_message(conversation_id, system_msg["content"])
    else:
        # Follow-up turn: reuse stored sources, no new retrieval
        assert session is not None and req.conversation_id is not None
        conversation_id = req.conversation_id
        if not session_manager.can_accept_message(conversation_id):
            raise HTTPException(
                status_code=400,
                detail="Maximale Anzahl an Rückfragen erreicht. Bitte starte eine neue Unterhaltung.",
            )
        results = session.sources

    session_manager.add_user_message(conversation_id, req.message)
    messages = session_manager.get_messages(conversation_id)
    answer = generator.chat(messages, temperature=req.temperature)
    session_manager.add_assistant_message(conversation_id, answer)

    # Build history (user/assistant messages only, exclude system)
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in session_manager.get_messages(conversation_id)
        if m["role"] != "system"
    ]

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=[SearchResultItem.from_retrieval_result(r) for r in results],
        history=history,
        remaining_followups=session_manager.remaining_followups(conversation_id),
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
