"""Application configuration via Pydantic Settings."""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    lancedb_fp: Path = Path("lancedb")

    # Embeddings
    embedding_model: str = "jinaai/jina-embeddings-v5-text-nano-retrieval"

    # LLM provider: "ollama" or "mistral"
    llm_provider: str = "ollama"
    generation_model: str = "qwen3:30b"
    ollama_url: str = "http://localhost:11434"
    mistral_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("MISTRAL_API_KEY"),
    )
    mistral_model: str = "mistral-small-latest"

    # Generation parameters
    default_temperature: float = 0.7
    default_max_tokens: int = 2048

    # Retrieval & reranking
    reranker_model: str = "Alibaba-NLP/gte-multilingual-reranker-base"
    rerank_top_k: int = 5
    retrieval_oversample_factor: int = 4
    min_similarity_score: float = 0.3

    # Chat sessions
    chat_session_ttl: int = 1800
    chat_max_followups: int = 3

    # Tavily web search
    tavily_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("TAVILY_API_KEY"),
    )
    tavily_search_depth: str = "basic"
    websearch_enabled: bool = False
    websearch_max_results: int = 5
    websearch_site_filters: list[str] = Field(
        default_factory=lambda: [
            "allesmuenster.de",
            "am-hawerkamp.de",
            "amnesty-muenster-osnabrueck.de",
            "ankommenapp.de",
            "arbeitsagentur.de",
            "awo-msl-re.de",
            "bamf.de",
            "berlitz.com",
            "bezreg-muenster.de",
            "bildungsinstitut.de",
            "bistum-muenster.de",
            "caritas-ms.de",
            "clemenshospital.de",
            "de.wikipedia.org",
            "deutschwerkstatt.de",
            "di-muenster.de",
            "diakonie-muenster.de",
            "digitalhub.ms",
            "drk-muenster.de",
            "fairteilbar-muenster.de",
            "fh-muenster.de",
            "freiwilligenagentur-muenster.de",
            "geba-muenster.de",
            "gesundheit-mehrsprachig.de",
            "gewaltschutz-muenster.de",
            "ggua.de",
            "gla.bildungsinstitut.de",
            "hebammennetzwerk-muensterland.de",
            "hjk-muenster.de",
            "hwk-muenster.de",
            "ihk.de",
            "integrationsrat-muenster.de",
            "iq-nrw-west.de",
            "katho-nrw.de",
            "kliniken.de",
            "kompanera.de",
            "kvwl.de",
            "landwirtschaftskammer.de",
            "lebenshilfe-muenster.de",
            "lwl-klinik-muenster.de",
            "muenster-fast-umsonst.de",
            "muenster-geht-aus.de",
            "muenster.polizei.nrw",
            "muensterland.com",
            "muensterzukunft.de",
            "nadann.de",
            "nebenan.de",
            "paritaetischer-muenster.de",
            "raphaelsklinik.de",
            "seht-muenster.de",
            "ssb.ms",
            "stadt-muenster.de",
            "stadtwerke-muenster.de",
            "starthilfe-muenster.de",
            "stw-muenster.de",
            "theater-muenster.com",
            "ukm.de",
            "uni-muenster.de",
            "verbraucherzentrale.nrw",
            "wipdaf-deutschkurse-muenster.de",
            "wochenmarkt-muenster.de",
            "www1.wdr.de",
            "www2.lwl.org",
            "zanzu.de",
        ]
    )
