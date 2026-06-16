"""Application configuration via Pydantic Settings."""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from muenster4you.websearch import SearchDepth


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
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
    rerank_top_k: int = 5
    retrieval_oversample_factor: int = 4
    min_similarity_score: float = 0.3

    # Chat sessions
    chat_session_ttl: int = 1800
    chat_max_followups: int = 3

    # Tavily web search
    tavily_api_key: str = Field(validation_alias=AliasChoices("TAVILY_API_KEY"))
    # Tavily search depth — trades credits/latency for recall:
    #   "basic"      (1 credit) — one NLP summary per URL, balanced.
    #   "advanced"   (2 credits) — multiple snippets per URL, highest relevance.
    #   "fast"       (1 credit) — multiple snippets, lower latency.
    #   "ultra-fast" (1 credit) — one summary, lowest latency.
    # https://docs.tavily.com/documentation/api-reference/endpoint/search
    tavily_search_depth: SearchDepth = "basic"
    websearch_max_results: int = 5
    # Tavily `include_domains` honors path prefixes, so curated paths from
    # muenster-urls.txt are preserved verbatim instead of being collapsed to
    # bare hostnames. Keep this list in sync with muenster-urls.txt.
    websearch_site_filters: frozenset[str] = Field(
        default_factory=lambda: frozenset({
            "allesmuenster.de",
            "am-hawerkamp.de",
            "amnesty-muenster-osnabrueck.de/gruppe/asylgruppe-muenster",
            "ankommenapp.de/APP/DE/Startseite/startseite-node.html",
            "arbeitsagentur.de/vor-ort/ahlen-muenster",
            "awo-msl-re.de/einrichtung/migrationsberatung-fuer-erwachsene-zuwandererinnen-5",
            "bamf.de/DE/Themen/Integration",
            "bamf.de/SharedDocs/Anlagen/DE/Integration/WillkommenDeutschland/willkommen-in-deutschland.html",
            "bezreg-muenster.de/themen/gesundheit-und-soziales/flucht-und-migration",
            "bildungsinstitut.de",
            "caritas-ms.de",
            "clemenshospital.de",
            "deutschwerkstatt.de",
            "di-muenster.de",
            "diakonie-muenster.de",
            "drk-muenster.de",
            "fairteilbar-muenster.de",
            "fh-muenster.de/internationaloffice",
            "freiwilligenagentur-muenster.de",
            "geba-muenster.de",
            "gesundheit-mehrsprachig.de",
            "gewaltschutz-muenster.de",
            "ggua.de/startseite",
            "hebammennetzwerk-muensterland.de/de",
            "hjk-muenster.de",
            "hwk-muenster.de/de/ausbildung/ausbildungsbetriebe/willkommenslotsen",
            "ihk.de/nordwestfalen/bildung/fachkraeftesicherung/fluechtlinge-3604362",
            "integrationsrat-muenster.de",
            "iq-nrw-west.de/info-beratungsstellen",
            "katho-nrw.de/studium/studieninteressierte-mit-fluchthintergrund/muenster",
            "kliniken.de/krankenhaus/deutschland/ort/münster",
            "kompanera.de",
            "lebenshilfe-muenster.de/de",
            "lwl-klinik-muenster.de/de",
            "lwl.org/de",
            "muenster-fast-umsonst.de",
            "muenster-geht-aus.de",
            "muenster.polizei.nrw",
            "muensterland.com",
            "muensterzukunft.de",
            "nadann.de",
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
            "uni-muenster.de/InternationalOffice",
            "verbraucherzentrale.nrw/beratungsstellen/muenster",
            "wipdaf-deutschkurse-muenster.de",
            "wochenmarkt-muenster.de",
            "zanzu.de/de",
        })
    )
