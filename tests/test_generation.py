from dataclasses import dataclass, field

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from muenster4you.rag.generation import (
    ERROR_MESSAGE,
    RAGGenerator,
    format_context,
    split_conversation,
)
from muenster4you.types import RetrievalResult, RetrievalSource

WIKI_DOC = RetrievalResult(
    content="Der Aasee ist ein See im Süden von Münster.",
    score=0.9,
    source=RetrievalSource.WIKI,
    url="https://wiki.example/index.php/Aasee_(See)",
)
WEB_DOC = RetrievalResult(
    content="Öffnungszeiten des Bürgerbüros.",
    score=0.8,
    source=RetrievalSource.WEBSEARCH,
    url="https://stadt-muenster.de/buergerbuero",
)


@dataclass
class _SpyModel:
    """Wraps a FunctionModel that answers with `reply` and records every request."""

    reply: str = "Antwort"
    requests: list[list[ModelMessage]] = field(default_factory=list)
    settings: list[dict] = field(default_factory=list)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append(messages)
        self.settings.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart(content=self.reply)])

    def model(self) -> FunctionModel:
        return FunctionModel(self._respond, model_name="spy")


def _generator(model, temperature: float = 0.7, max_tokens: int = 2048) -> RAGGenerator:
    return RAGGenerator(model, default_temperature=temperature, default_max_tokens=max_tokens)


def _user_prompt(request: ModelMessage) -> str:
    assert isinstance(request, ModelRequest)
    (part,) = (p for p in request.parts if isinstance(p, UserPromptPart))
    assert isinstance(part.content, str)
    return part.content


def test_format_context_labels_wiki_and_web_sources():
    context = format_context([WIKI_DOC, WEB_DOC])

    assert "[Dokument 1 (Wiki): Aasee (See)]" in context
    assert "[Dokument 2 (Web): https://stadt-muenster.de/buergerbuero]" in context
    assert "URL: https://stadt-muenster.de/buergerbuero" in context


def test_format_context_truncates_long_content():
    long_doc = RetrievalResult(
        content="x" * 2500, score=0.5, source=RetrievalSource.WIKI, url="/wiki/Lang"
    )

    context = format_context([long_doc])

    assert "x" * 2000 + "..." in context
    assert "x" * 2001 not in context


def test_split_conversation_maps_roles_to_history_and_prompt():
    history, prompt = split_conversation(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "erste Frage"},
            {"role": "assistant", "content": "erste Antwort"},
            {"role": "user", "content": "zweite Frage"},
        ]
    )

    assert prompt == "zweite Frage"
    assert [type(m) for m in history] == [ModelRequest, ModelResponse]
    assert [type(p) for p in history[0].parts] == [SystemPromptPart, UserPromptPart]
    assert history[1].parts == [TextPart(content="erste Antwort")]


def test_split_conversation_keeps_lone_system_message_in_history():
    history, prompt = split_conversation(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "Frage"}]
    )

    assert prompt == "Frage"
    (request,) = history
    assert isinstance(request, ModelRequest)
    assert [(type(p), p.content) for p in request.parts] == [(SystemPromptPart, "sys")]


def test_split_conversation_rejects_trailing_assistant_message():
    with pytest.raises(ValueError):
        split_conversation([{"role": "assistant", "content": "x"}])


def test_generate_embeds_context_and_query_in_prompt():
    spy = _SpyModel(reply="Der Aasee liegt im Süden.")

    answer = _generator(spy.model()).generate("Wo liegt der Aasee?", [WIKI_DOC])

    assert answer == "Der Aasee liegt im Süden."
    prompt = _user_prompt(spy.requests[0][-1])
    assert "Benutzerfrage: Wo liegt der Aasee?" in prompt
    assert "[Dokument 1 (Wiki): Aasee (See)]" in prompt
    assert WIKI_DOC.content in prompt


def test_generate_defaults_to_german_and_accepts_other_languages():
    spy = _SpyModel()
    generator = _generator(spy.model())

    generator.generate("q", [])
    generator.generate("q", [], language="Arabisch")

    assert "Antworte auf Deutsch." in _user_prompt(spy.requests[0][-1])
    assert "Antworte auf Arabisch." in _user_prompt(spy.requests[1][-1])


def test_system_message_carries_language_and_history_rule():
    message = _generator(TestModel()).build_system_message([WIKI_DOC], language="Türkisch")

    assert message["role"] == "system"
    assert "Antworte auf Türkisch." in message["content"]
    assert "Gesprächsverlauf" in message["content"]
    assert WIKI_DOC.content in message["content"]


def test_generate_uses_defaults_and_per_call_overrides():
    spy = _SpyModel()
    generator = _generator(spy.model(), temperature=0.2, max_tokens=100)

    generator.generate("q", [])
    generator.generate("q", [], temperature=0.9)
    generator.generate("q", [], max_tokens=5)

    assert spy.settings[0] == {"temperature": 0.2, "max_tokens": 100}
    assert spy.settings[1] == {"temperature": 0.9, "max_tokens": 100}
    assert spy.settings[2] == {"temperature": 0.2, "max_tokens": 5}


def test_generate_returns_error_message_when_model_fails():
    def explode(_messages, _info):
        raise RuntimeError("kaputt")

    answer = _generator(FunctionModel(explode)).generate("q", [])

    assert answer == ERROR_MESSAGE.format(error="kaputt")


async def test_chat_sends_history_and_prompt():
    spy = _SpyModel(reply="Zweite Antwort")
    generator = _generator(spy.model())
    messages = [
        generator.build_system_message([WIKI_DOC]),
        {"role": "user", "content": "erste Frage"},
        {"role": "assistant", "content": "erste Antwort"},
        {"role": "user", "content": "zweite Frage"},
    ]

    answer = await generator.chat(messages, temperature=0.1)

    assert answer == "Zweite Antwort"
    (request,) = spy.requests
    assert [type(m) for m in request] == [ModelRequest, ModelResponse, ModelRequest]
    system_part = request[0].parts[0]
    assert isinstance(system_part, SystemPromptPart)
    assert WIKI_DOC.content in system_part.content
    assert _user_prompt(request[0]) == "erste Frage"
    assert _user_prompt(request[2]) == "zweite Frage"
    assert spy.settings[0]["temperature"] == 0.1


async def test_chat_returns_error_message_when_model_fails():
    def explode(_messages, _info):
        raise RuntimeError("kaputt")

    answer = await _generator(FunctionModel(explode)).chat([{"role": "user", "content": "q"}])

    assert answer == ERROR_MESSAGE.format(error="kaputt")


def test_generate_stream_yields_deltas():
    chunks = list(
        _generator(TestModel(custom_output_text="Der Aasee")).generate_stream("q", [WIKI_DOC])
    )

    assert "".join(chunks) == "Der Aasee"
    assert len(chunks) >= 1
