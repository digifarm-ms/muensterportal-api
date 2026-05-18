"""Generation layer using OpenAI-compatible API for RAG responses."""

from typing import Iterator, List
from urllib.parse import unquote

from openai import OpenAI

from ..config import AppConfig
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


class RAGGenerator:
    """Generator for RAG responses using OpenAI-compatible APIs (Ollama or Mistral)."""

    def __init__(self, config: AppConfig):
        self.provider = config.llm_provider
        self.default_temperature = config.default_temperature
        self.default_max_tokens = config.default_max_tokens

        if self.provider == "mistral":
            self.model_name = config.mistral_model
            self.client = OpenAI(
                base_url="https://api.mistral.ai/v1",
                api_key=config.mistral_api_key,
            )
        else:
            self.model_name = config.generation_model
            self.client = OpenAI(
                base_url=f"{config.ollama_url}/v1",
                api_key="ollama",
            )

        print(
            f"RAG Generator initialized with provider={self.provider}, model={self.model_name}"
        )

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []

        for i, result in enumerate(results, 1):
            content = result.content
            if len(content) > 2000:
                content = content[:2000] + "..."

            is_web = result.source == RetrievalSource.WEBSEARCH
            source_label = "Web" if is_web else "Wiki"
            if is_web:
                title = result.url
            else:
                title = unquote(result.url.rsplit("/", 1)[-1]).replace("_", " ")
            header = f"[Dokument {i} ({source_label}): {title}]"

            if is_web and result.url:
                header += f"\nURL: {result.url}"

            context_parts.append(f"{header}\n{content}\n")

        return "\n".join(context_parts)

    def _call(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Single-turn generation: send a prompt, get a text response."""
        if self.provider == "mistral":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        else:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            return response.output_text

    def _call_chat(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> str:
        """Multi-turn generation: send a message history, get a text response."""
        if self.provider == "mistral":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        else:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            return response.output_text

    def _call_stream(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Iterator[str]:
        """Streaming single-turn generation."""
        if self.provider == "mistral":
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            stream = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                stream=True,
            )
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    def generate(
        self,
        query: str,
        context_docs: List[RetrievalResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate an answer using retrieved context."""
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        context = self._format_context(context_docs)
        prompt = GERMAN_RAG_PROMPT.format(context=context, query=query)

        try:
            return self._call(prompt, temperature, max_tokens)
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Entschuldigung, es gab einen Fehler bei der Generierung der Antwort: {e}"

    def build_system_message(self, context_docs: List[RetrievalResult]) -> dict:
        """Build a system message with RAG context for multi-turn chat."""
        context = self._format_context(context_docs)
        content = GERMAN_RAG_CHAT_SYSTEM_PROMPT.format(context=context)
        return {"role": "system", "content": content}

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response using a message history (multi-turn chat)."""
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        try:
            return self._call_chat(messages, temperature, max_tokens)
        except Exception as e:
            print(f"Error generating chat response: {e}")
            return f"Entschuldigung, es gab einen Fehler bei der Generierung der Antwort: {e}"

    def generate_stream(
        self,
        query: str,
        context_docs: List[RetrievalResult],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Generate an answer with streaming (for real-time display)."""
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        context = self._format_context(context_docs)
        prompt = GERMAN_RAG_PROMPT.format(context=context, query=query)

        try:
            yield from self._call_stream(prompt, temperature, max_tokens)
        except Exception as e:
            print(f"Error generating response: {e}")
            yield f"Entschuldigung, es gab einen Fehler bei der Generierung der Antwort: {e}"
