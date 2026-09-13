"""HTTP API package.

`legacy` holds the current single-module app while it is being refactored into
`routes` (and friends). Everything public is re-exported here so
`muenster4you.api:app`, `muenster4you.api:main` and the existing tests keep
working unchanged.
"""

from muenster4you.api.legacy import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConfigDep,
    EmbeddingModelDep,
    GeneratorDep,
    ModelDep,
    OrchestratorDep,
    RerankerDep,
    RetrieverDep,
    SearchResponse,
    SessionManagerDep,
    SourceItem,
    WebSearcherDep,
    app,
    get_config,
    get_embedding_model,
    get_generator,
    get_model,
    get_orchestrator,
    get_reranker,
    get_retriever,
    get_session_manager,
    get_web_searcher,
    main,
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ConfigDep",
    "EmbeddingModelDep",
    "GeneratorDep",
    "ModelDep",
    "OrchestratorDep",
    "RerankerDep",
    "RetrieverDep",
    "SearchResponse",
    "SessionManagerDep",
    "SourceItem",
    "WebSearcherDep",
    "app",
    "get_config",
    "get_embedding_model",
    "get_generator",
    "get_model",
    "get_orchestrator",
    "get_reranker",
    "get_retriever",
    "get_session_manager",
    "get_web_searcher",
    "main",
]
