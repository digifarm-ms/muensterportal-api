from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from muenster4you.lancedb import EMBEDDING_DIM


@dataclass
class FakeEmbedder:
    """Returns a predetermined vector regardless of input."""

    rng: Generator

    def encode(self, _text: str) -> NDArray[np.float32]:
        return self.rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
