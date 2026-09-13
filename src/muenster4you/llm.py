"""Builds the pydantic-ai model selected by configuration."""

from collections.abc import Callable

from pydantic_ai.models import Model
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_ai.providers.ollama import OllamaProvider

from muenster4you.config import AppConfig, LLMProvider


def _mistral(config: AppConfig) -> Model:
    return MistralModel(
        config.mistral_model,
        provider=MistralProvider(api_key=config.mistral_api_key),
    )


def _ollama(config: AppConfig) -> Model:
    return OllamaModel(
        config.generation_model,
        provider=OllamaProvider(base_url=f"{config.ollama_url}/v1"),
    )


MODEL_FACTORIES: dict[LLMProvider, Callable[[AppConfig], Model]] = {
    "mistral": _mistral,
    "ollama": _ollama,
}


def build_model(config: AppConfig) -> Model:
    return MODEL_FACTORIES[config.llm_provider](config)
