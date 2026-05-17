"""FastAPI application with RAG-powered search and query endpoints."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from muenster4you.config import AppConfig
from muenster4you.embedder import SentenceTransformerEmbedder
from muenster4you.retriever import LanceDBRetriever, RetrievalResult


# --- Dependencies ---


@lru_cache
def get_config() -> AppConfig:
    return AppConfig()


ConfigDep = Annotated[AppConfig, Depends(get_config)]


@lru_cache
def get_retriever(config: ConfigDep) -> LanceDBRetriever:
    model = SentenceTransformer(model_name_or_path=config.embedding_model)
    embedder = SentenceTransformerEmbedder(model=model)
    return LanceDBRetriever(db_path=config.lancedb_fp, embedder=embedder)


RetrieverDep = Annotated[LanceDBRetriever, Depends(get_retriever)]


# --- App ---


app = FastAPI(title="Muenster4You API", version="0.1.0")


# --- Models ---


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    query: str | None
    results: list[RetrievalResult] = []
    message: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[RetrievalResult]


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
    sources: list[RetrievalResult]
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
    query: str = Query(..., min_length=3, description="Search query string"),
    top_k: int = Query(5, ge=1, le=20),
) -> SearchResponse:
    results = retriever.search(query, top_k=top_k)

    return SearchResponse(
        query=query,
        results=results,
    )


# @app.post("/query")
# async def query(
#     retriever: RetrieverDep,
#     generator: GeneratorDep,
#     req: QueryRequest,
# ) -> QueryResponse:
#     results = retriever.search(req.question, limit=req.top_k)
#     answer = generator.generate(req.question, results, temperature=req.temperature)

#     return QueryResponse(
#         question=req.question,
#         answer=answer,
#         sources=[SearchResultItem.from_retrieval_result(r) for r in results],
#     )


# @app.post("/chat")
# async def chat(
#     retriever: RetrieverDep,
#     generator: GeneratorDep,
#     session_manager: SessionManagerDep,
#     req: ChatRequest,
# ) -> ChatResponse:
#     session_manager.cleanup_expired()

#     is_new = True
#     session = None
#     if req.conversation_id:
#         session = session_manager.get_session(req.conversation_id)
#         if session is not None:
#             is_new = False

#     if is_new:
#         # First turn: run RAG retrieval and create session
#         results = retriever.search(req.message, limit=req.top_k)
#         conversation_id = session_manager.create_session(sources=results)
#         system_msg = generator.build_system_message(results)
#         session_manager.set_system_message(conversation_id, system_msg["content"])
#     else:
#         # Follow-up turn: reuse stored sources, no new retrieval
#         assert session is not None and req.conversation_id is not None
#         conversation_id = req.conversation_id
#         if not session_manager.can_accept_message(conversation_id):
#             raise HTTPException(
#                 status_code=400,
#                 detail="Maximale Anzahl an Rückfragen erreicht. Bitte starte eine neue Unterhaltung.",
#             )
#         results = session.sources

#     session_manager.add_user_message(conversation_id, req.message)
#     messages = session_manager.get_messages(conversation_id)
#     answer = generator.chat(messages, temperature=req.temperature)
#     session_manager.add_assistant_message(conversation_id, answer)

#     # Build history (user/assistant messages only, exclude system)
#     history = [
#         ChatMessage(role=m["role"], content=m["content"])
#         for m in session_manager.get_messages(conversation_id)
#         if m["role"] != "system"
#     ]

#     return ChatResponse(
#         conversation_id=conversation_id,
#         answer=answer,
#         sources=[SearchResultItem.from_retrieval_result(r) for r in results],
#         history=history,
#         remaining_followups=session_manager.remaining_followups(conversation_id),
#     )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
