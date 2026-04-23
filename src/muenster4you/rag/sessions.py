"""In-memory chat session management for multi-turn conversations."""

import time
import uuid
from dataclasses import dataclass, field

from .config import config
from .retrieval import RetrievalResult


@dataclass
class ConversationSession:
    """A single chat conversation with its state."""

    id: str
    messages: list[dict] = field(default_factory=list)
    sources: list[RetrievalResult] = field(default_factory=list)
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class ChatSessionManager:
    """Manages in-memory chat sessions with TTL expiry."""

    def __init__(
        self,
        ttl: int | None = None,
        max_followups: int | None = None,
    ):
        self._sessions: dict[str, ConversationSession] = {}
        self._ttl = ttl if ttl is not None else config.chat_session_ttl
        self._max_followups = (
            max_followups if max_followups is not None else config.chat_max_followups
        )

    def create_session(self, sources: list[RetrievalResult]) -> str:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = ConversationSession(
            id=session_id,
            sources=sources,
        )
        return session_id

    def get_session(self, session_id: str) -> ConversationSession | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.last_active > self._ttl:
            del self._sessions[session_id]
            return None
        return session

    def set_system_message(self, session_id: str, content: str) -> None:
        session = self._sessions[session_id]
        system_msg = {"role": "system", "content": content}
        if session.messages and session.messages[0]["role"] == "system":
            session.messages[0] = system_msg
        else:
            session.messages.insert(0, system_msg)

    def add_user_message(self, session_id: str, content: str) -> None:
        session = self._sessions[session_id]
        session.messages.append({"role": "user", "content": content})
        session.turn_count += 1
        session.last_active = time.time()

    def add_assistant_message(self, session_id: str, content: str) -> None:
        session = self._sessions[session_id]
        session.messages.append({"role": "assistant", "content": content})

    def get_messages(self, session_id: str) -> list[dict]:
        return self._sessions[session_id].messages

    def remaining_followups(self, session_id: str) -> int:
        session = self._sessions[session_id]
        return max(0, self._max_followups - (session.turn_count - 1))

    def can_accept_message(self, session_id: str) -> bool:
        session = self._sessions[session_id]
        max_turns = 1 + self._max_followups
        return session.turn_count < max_turns

    def cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
