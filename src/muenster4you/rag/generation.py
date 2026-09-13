"""Generation layer for RAG responses."""

from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import unquote

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from ..language import DEFAULT_LANGUAGE
from ..types import RetrievalResult, RetrievalSource

SYSTEM_PROMPT = """Du bist der Assistent von Münster4You. Du hilfst Menschen, die in Münster leben oder neu ankommen, mit Informationen zu Behörden, Beratung, Sprache, Arbeit, Wohnen, Gesundheit und Freizeit in Münster.

Regeln:
- Antworte auf {language}. Deutsche Eigennamen (Behörden, Einrichtungen, Straßen, Programme) bleiben unverändert im Original.
- Nutze ausschließlich die Kontext-Dokumente. Erfinde keine Adressen, Telefonnummern, E-Mail-Adressen, Links, Öffnungszeiten oder Fristen; gib solche Angaben genau so wieder, wie sie im Dokument stehen.
- Wenn die Dokumente die Frage nicht beantworten, sage das klar und nenne höchstens eine Stelle aus den Dokumenten, die weiterhelfen könnte.
- Belege jede Aussage mit der Nummer des Dokuments in eckigen Klammern, zum Beispiel [1] oder [2][4]. Keine anderen Zitierformen, keine Links.
- Antworte kurz und konkret, höchstens etwa 120 Wörter.
- Berücksichtige den bisherigen Gesprächsverlauf.

Kontext-Dokumente:
{context}"""

ERROR_MESSAGE = "Entschuldigung, es gab einen Fehler bei der Generierung der Antwort: {error}"


@dataclass(frozen=True)
class RunDeps:
    context_docs: list[RetrievalResult]
    language: str = DEFAULT_LANGUAGE


def format_context(results: list[RetrievalResult]) -> str:
    context_parts = []

    for i, result in enumerate(results, 1):
        content = result.content
        if len(content) > 2000:
            content = content[:2000] + "..."

        is_web = result.source == RetrievalSource.WEBSEARCH
        source_label = "Web" if is_web else "Wiki"
        title = result.url if is_web else unquote(result.url.rsplit("/", 1)[-1]).replace("_", " ")
        header = f"[Dokument {i} ({source_label}): {title}]"

        if is_web and result.url:
            header += f"\nURL: {result.url}"

        context_parts.append(f"{header}\n{content}\n")

    return "\n".join(context_parts)


def render_instructions(ctx: RunContext[RunDeps]) -> str:
    return SYSTEM_PROMPT.format(
        language=ctx.deps.language, context=format_context(ctx.deps.context_docs)
    )


def split_conversation(messages: list[dict]) -> tuple[list[ModelMessage], str]:
    """Split user/assistant dicts into pydantic-ai history plus the final user prompt."""
    *earlier, last = messages
    if last["role"] != "user":
        raise ValueError("conversation must end with a user message")

    history: list[ModelMessage] = []
    for message in earlier:
        match message["role"]:
            case "user":
                history.append(ModelRequest(parts=[UserPromptPart(content=message["content"])]))
            case "assistant":
                history.append(ModelResponse(parts=[TextPart(content=message["content"])]))
            case role:
                raise ValueError(f"unknown role: {role}")
    return history, last["content"]


def _settings(temperature: float | None, max_tokens: int | None) -> ModelSettings | None:
    settings: ModelSettings = {}
    if temperature is not None:
        settings["temperature"] = temperature
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    return settings or None


class RAGGenerator:
    def __init__(self, model: Model, default_temperature: float, default_max_tokens: int):
        self.model_name = model.model_name
        self._agent = Agent(
            model,
            deps_type=RunDeps,
            instructions=render_instructions,
            model_settings=ModelSettings(
                temperature=default_temperature, max_tokens=default_max_tokens
            ),
        )

    def generate(
        self,
        query: str,
        context_docs: list[RetrievalResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
        language: str = DEFAULT_LANGUAGE,
    ) -> str:
        try:
            result = self._agent.run_sync(
                query,
                deps=RunDeps(context_docs, language),
                model_settings=_settings(temperature, max_tokens),
            )
            return result.output
        except Exception as e:
            return ERROR_MESSAGE.format(error=e)

    async def chat(
        self,
        messages: list[dict],
        context_docs: list[RetrievalResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
        language: str = DEFAULT_LANGUAGE,
    ) -> str:
        history, prompt = split_conversation(messages)
        try:
            result = await self._agent.run(
                prompt,
                deps=RunDeps(context_docs, language),
                message_history=history,
                model_settings=_settings(temperature, max_tokens),
            )
            return result.output
        except Exception as e:
            return ERROR_MESSAGE.format(error=e)

    def generate_stream(
        self,
        query: str,
        context_docs: list[RetrievalResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
        language: str = DEFAULT_LANGUAGE,
    ) -> Iterator[str]:
        try:
            result = self._agent.run_stream_sync(
                query,
                deps=RunDeps(context_docs, language),
                model_settings=_settings(temperature, max_tokens),
            )
            yield from result.stream_text(delta=True)
        except Exception as e:
            yield ERROR_MESSAGE.format(error=e)
