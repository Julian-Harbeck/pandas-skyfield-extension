import numpy as np
import pytest


@pytest.fixture
def test_array_1d() -> np.ndarray:
    return np.arange(1, 4)


@pytest.fixture
def test_array_2d() -> np.ndarray:
    return np.arange(1, 13).reshape((4, 3)).T
