"""Generation layer using Ollama for RAG responses."""

from typing import Iterator, List

import ollama

from .config import config
from .retrieval import RetrievalResult


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
    """Generator for RAG responses using Ollama."""

    def __init__(
        self,
        model_name: str = None,
        ollama_url: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize the RAG generator.

        Args:
            model_name: Ollama model name
            ollama_url: Ollama server URL
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name or config.generation_model
        self.ollama_url = ollama_url or config.ollama_url
        self.default_temperature = temperature or config.default_temperature
        self.default_max_tokens = max_tokens or config.default_max_tokens

        # Configure ollama client
        self.client = ollama.Client(host=self.ollama_url)

        print(f"RAG Generator initialized with model: {self.model_name}")

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieved documents into context string.

        Args:
            results: List of retrieval results

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            # Truncate very long content
            content = result.content_text
            if len(content) > 2000:
                content = content[:2000] + "..."

            # Add source label (Wiki or Web)
            source_label = "Web" if result.source == "web" else "Wiki"
            header = f"[Dokument {i} ({source_label}): {result.page_title}]"

            # Add URL for web sources
            if result.source == "web" and result.source_url:
                header += f"\nURL: {result.source_url}"

            context_parts.append(f"{header}\n{content}\n")

        return "\n".join(context_parts)

    def generate(
        self,
        query: str,
        context_docs: List[RetrievalResult],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate an answer using retrieved context.

        Args:
            query: User question
            context_docs: Retrieved documents
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated answer
        """
        if temperature is None:
            temperature = self.default_temperature

        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # Format context
        context = self._format_context(context_docs)

        # Build prompt
        prompt = GERMAN_RAG_PROMPT.format(context=context, query=query)

        try:
            # Call Ollama
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )

            return response["response"]

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return f"Entschuldigung, es gab einen Fehler bei der Generierung der Antwort: {str(e)}"

    def build_system_message(self, context_docs: List[RetrievalResult]) -> dict:
        """Build a system message with RAG context for multi-turn chat."""
        context = self._format_context(context_docs)
        content = GERMAN_RAG_CHAT_SYSTEM_PROMPT.format(context=context)
        return {"role": "system", "content": content}

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Generate a response using the ollama chat API with a message history."""
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            return response.message.content

        except Exception as e:
            error_msg = f"Error generating chat response: {str(e)}"
            print(error_msg)
            return f"Entschuldigung, es gab einen Fehler bei der Generierung der Antwort: {str(e)}"

    def generate_stream(
        self,
        query: str,
        context_docs: List[RetrievalResult],
        temperature: float = None,
        max_tokens: int = None
    ) -> Iterator[str]:
        """
        Generate an answer with streaming (for real-time display).

        Args:
            query: User question
            context_docs: Retrieved documents
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they are generated
        """
        if temperature is None:
            temperature = self.default_temperature

        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # Format context
        context = self._format_context(context_docs)

        # Build prompt
        prompt = GERMAN_RAG_PROMPT.format(context=context, query=query)

        try:
            # Call Ollama with streaming
            stream = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )

            for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            yield f"Entschuldigung, es gab einen Fehler bei der Generierung der Antwort: {str(e)}"


if __name__ == "__main__":
    # Test generation with dummy context
    from .retrieval import RetrievalResult

    dummy_docs = [
        RetrievalResult(
            page_id=1,
            page_title="Hofläden",
            content_text="In Münster gibt es viele Hofläden, die frisches Gemüse und Obst verkaufen. "
                         "Besonders beliebt sind die Hofläden in Handorf und Hiltrup.",
            similarity_score=0.85,
            page_len=200
        ),
        RetrievalResult(
            page_id=2,
            page_title="Freizeit",
            content_text="Münster bietet viele Freizeitmöglichkeiten wie Radfahren am Aasee, "
                         "Besuche im Allwetterzoo oder Spaziergänge in der Altstadt.",
            similarity_score=0.72,
            page_len=150
        ),
    ]

    generator = RAGGenerator()

    test_query = "Wo kann ich frisches Gemüse kaufen?"
    print(f"Query: {test_query}\n")
    print("Generating answer...")

    answer = generator.generate(test_query, dummy_docs)
    print(f"\nAnswer:\n{answer}")
