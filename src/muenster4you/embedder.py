from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import get_device_name


class TextEmbedder(Protocol):
    """Turns a text into a normalized embedding vector.

    Returns a float32 ndarray — LanceDB's search casts to this internally,
    so matching it avoids an extra allocation.
    """

    def encode(self, text: str) -> NDArray[np.float32]: ...


@dataclass
class SentenceTransformerEmbedder(TextEmbedder):
    model: SentenceTransformer

    def encode(self, text: str) -> NDArray[np.float32]:
        return self.model.encode(text, prompt_name="query", convert_to_numpy=True)


def load_embedder(model_name: str, device: str | None = None) -> SentenceTransformer:
    """Load the embedding model, in float32 when it runs on a CPU.

    The jina model ships with bfloat16 weights. CPUs without native bf16
    matmul (anything before AVX512-BF16/AMX) emulate it and get 7-100x slower.
    """
    device = device or get_device_name()
    model_kwargs = {"dtype": torch.float32} if device == "cpu" else {}
    return SentenceTransformer(
        model_name, trust_remote_code=True, device=device, model_kwargs=model_kwargs
    )
