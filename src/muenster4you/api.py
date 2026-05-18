"""FastAPI application with RAG-powered search and chat endpoints."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

from muenster4you.config import AppConfig
from muenster4you.embedder import SentenceTransformerEmbedder
from muenster4you.rag.generation import RAGGenerator
from muenster4you.rag.sessions import ChatSessionManager
from muenster4you.reranker import CrossEncoderReranker, Reranker
from muenster4you.retrieval import RetrievalOrchestrator
from muenster4you.retriever import LanceDBRetriever
from muenster4you.types import RetrievalResult
from muenster4you.websearch import TavilySearcher


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


@lru_cache
def get_web_searcher(config: ConfigDep) -> TavilySearcher | None:
    if not config.websearch_enabled or not config.tavily_api_key:
        return None
    client = TavilyClient(api_key=config.tavily_api_key)
    return TavilySearcher(client=client, site_filters=config.websearch_site_filters)


WebSearcherDep = Annotated[TavilySearcher | None, Depends(get_web_searcher)]


@lru_cache
def get_reranker(config: ConfigDep) -> Reranker:
    return CrossEncoderReranker(model_id=config.reranker_model)


RerankerDep = Annotated[Reranker, Depends(get_reranker)]


def get_orchestrator(
    config: ConfigDep,
    retriever: RetrieverDep,
    web_searcher: WebSearcherDep,
    reranker: RerankerDep,
) -> RetrievalOrchestrator:
    return RetrievalOrchestrator(
        wiki_retriever=retriever,
        web_searcher=web_searcher,
        reranker=reranker,
        rerank_top_k=config.rerank_top_k,
        oversample_factor=config.retrieval_oversample_factor,
    )


OrchestratorDep = Annotated[RetrievalOrchestrator, Depends(get_orchestrator)]


@lru_cache
def get_generator(config: ConfigDep) -> RAGGenerator:
    return RAGGenerator(config)


GeneratorDep = Annotated[RAGGenerator, Depends(get_generator)]


@lru_cache
def get_session_manager(config: ConfigDep) -> ChatSessionManager:
    return ChatSessionManager(config)


SessionManagerDep = Annotated[ChatSessionManager, Depends(get_session_manager)]


# --- App ---


app = FastAPI(title="Muenster4You API", version="0.1.0")


# --- Models ---


class SearchResponse(BaseModel):
    query: str | None
    results: list[RetrievalResult] = []
    message: str | None = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
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
    orchestrator: OrchestratorDep,
    query: str = Query(..., min_length=3, description="Search query string"),
) -> SearchResponse:
    results = orchestrator.retrieve(query)
    return SearchResponse(query=query, results=results)


@app.post("/chat")
async def chat(
    orchestrator: OrchestratorDep,
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
        results = orchestrator.retrieve(req.message)
        conversation_id = session_manager.create_session(sources=results)
        system_msg = generator.build_system_message(results)
        session_manager.set_system_message(conversation_id, system_msg["content"])
    else:
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

    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in session_manager.get_messages(conversation_id)
        if m["role"] != "system"
    ]

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=results,
        history=history,
        remaining_followups=session_manager.remaining_followups(conversation_id),
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
