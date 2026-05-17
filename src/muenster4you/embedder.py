from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


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
        return self.model.encode(text, convert_to_numpy=True)
