"""Detects the user's language and maps it to the name used in the prompt."""

from fast_langdetect import detect

DEFAULT_LANGUAGE = "Deutsch"

# Languages the generation model answered fluently in the multilingual eval.
# Anything else falls back to German; the model degenerates in e.g. Kurdish,
# Pashto, Somali, Tigrinya and Amharic.
LANGUAGE_NAMES: dict[str, str] = {
    "de": "Deutsch",
    "en": "Englisch",
    "ar": "Arabisch",
    "tr": "Türkisch",
    "uk": "Ukrainisch",
    "ru": "Russisch",
    "fa": "Persisch",
    "pl": "Polnisch",
    "fr": "Französisch",
    "es": "Spanisch",
    "it": "Italienisch",
    "pt": "Portugiesisch",
    "nl": "Niederländisch",
    "ro": "Rumänisch",
    "bg": "Bulgarisch",
    "el": "Griechisch",
    "hr": "Kroatisch",
    "sr": "Serbisch",
    "bs": "Bosnisch",
    "sq": "Albanisch",
    "ur": "Urdu",
    "hi": "Hindi",
    "vi": "Vietnamesisch",
    "zh": "Chinesisch",
}

MIN_CONFIDENCE = 0.5


def detect_language(text: str, previous: str | None = None) -> str:
    """Return the prompt language name for `text`.

    Falls back to the conversation's previous language (or German) when the
    detector is unsure, which happens on very short messages like "Hallo".
    """
    result = detect(text, model="lite", k=1)
    if not result or float(result[0]["score"]) < MIN_CONFIDENCE:
        return previous or DEFAULT_LANGUAGE
    return LANGUAGE_NAMES.get(str(result[0]["lang"]), DEFAULT_LANGUAGE)
