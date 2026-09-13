import pytest

from muenster4you import language
from muenster4you.language import DEFAULT_LANGUAGE, detect_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Wo kann ich mich in Münster anmelden?", "Deutsch"),
        ("Where can I register my address in Münster?", "Englisch"),
        ("Münster'de nereye kayıt yaptırabilirim? Bürgerbüro nerede?", "Türkisch"),
        ("أين يمكنني تسجيل عنواني في مونستر؟", "Arabisch"),
        ("Де я можу зареєструватися в Мюнстері?", "Ukrainisch"),
        ("کجا می‌توانم در مونستر ثبت نام کنم؟", "Persisch"),
    ],
)
def test_detects_supported_languages(text: str, expected: str):
    assert detect_language(text) == expected


def test_short_message_falls_back_to_previous_language():
    assert detect_language("ok danke", previous="Englisch") == "Englisch"
    assert detect_language("Hallo") == DEFAULT_LANGUAGE


def test_unsupported_language_falls_back_to_german(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(language, "detect", lambda *_a, **_kw: [{"lang": "so", "score": 0.99}])

    assert detect_language("x", previous="Englisch") == DEFAULT_LANGUAGE
