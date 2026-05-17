import numpy as np
from numpy.random import Generator
import pytest


@pytest.fixture
def numpy_rng() -> Generator:
    return np.random.default_rng(seed=42)
