"""Generation layer for RAG responses."""

from collections.abc import Iterator
from urllib.parse import unquote

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from ..types import RetrievalResult, RetrievalSource

# German RAG prompt template (single-turn)
GERMAN_RAG_PROMPT = """Du bist ein hilfreicher Assistent für Informationen über die Stadt Münster in Deutschland. Deine Aufgabe ist es, Fragen basierend auf den bereitgestellten Dokumenten zu beantworten.

Kontext-Dokumente:
{context}

Benutzerfrage: {query}

Anweisungen:
- Antworte auf Deutsch
- Basiere deine Antwort ausschließlich auf den bereitgestellten Dokumenten
- Sei präzise und hilfreich
- Wenn die Information nicht in den Dokumenten zu finden ist, sage das ehrlich
- Zitiere relevante Dokumente wenn möglich (z.B. "Laut [Dokument: Titel]...")

Antwort:"""

# German RAG system prompt for multi-turn chat
GERMAN_RAG_CHAT_SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent für Informationen über die Stadt Münster in Deutschland. Deine Aufgabe ist es, Fragen basierend auf den bereitgestellten Dokumenten zu beantworten.

Anweisungen:
- Antworte auf Deutsch
- Basiere deine Antwort ausschließlich auf den bereitgestellten Dokumenten
- Sei präzise und hilfreich
- Wenn die Information nicht in den Dokumenten zu finden ist, sage das ehrlich
- Zitiere relevante Dokumente wenn möglich (z.B. "Laut [Dokument: Titel]...")
- Berücksichtige den bisherigen Gesprächsverlauf bei deinen Antworten

Kontext-Dokumente:
{context}"""

ERROR_MESSAGE = "Entschuldigung, es gab einen Fehler bei der Generierung der Antwort: {error}"


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


def split_conversation(messages: list[dict]) -> tuple[list[ModelMessage], str]:
    """Split role/content dicts into pydantic-ai history plus the final user prompt."""
    *earlier, last = messages
    if last["role"] != "user":
        raise ValueError("conversation must end with a user message")

    history: list[ModelMessage] = []
    pending: list[SystemPromptPart | UserPromptPart] = []
    for message in earlier:
        match message["role"]:
            case "system":
                pending.append(SystemPromptPart(content=message["content"]))
            case "user":
                pending.append(UserPromptPart(content=message["content"]))
                history.append(ModelRequest(parts=pending))
                pending = []
            case "assistant":
                history.append(ModelResponse(parts=[TextPart(content=message["content"])]))
            case role:
                raise ValueError(f"unknown role: {role}")
    if pending:
        history.append(ModelRequest(parts=pending))
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
    ) -> str:
        prompt = GERMAN_RAG_PROMPT.format(context=format_context(context_docs), query=query)
        try:
            result = self._agent.run_sync(prompt, model_settings=_settings(temperature, max_tokens))
            return result.output
        except Exception as e:
            return ERROR_MESSAGE.format(error=e)

    def build_system_message(self, context_docs: list[RetrievalResult]) -> dict:
        content = GERMAN_RAG_CHAT_SYSTEM_PROMPT.format(context=format_context(context_docs))
        return {"role": "system", "content": content}

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        history, prompt = split_conversation(messages)
        try:
            result = await self._agent.run(
                prompt,
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
    ) -> Iterator[str]:
        prompt = GERMAN_RAG_PROMPT.format(context=format_context(context_docs), query=query)
        try:
            result = self._agent.run_stream_sync(
                prompt, model_settings=_settings(temperature, max_tokens)
            )
            yield from result.stream_text(delta=True)
        except Exception as e:
            yield ERROR_MESSAGE.format(error=e)
