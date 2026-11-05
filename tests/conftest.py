import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RTOL = 1e-4
ATOL = 1e-4


def assert_close(actual, expected, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float64),
                               np.asarray(expected, dtype=np.float64), rtol=rtol, atol=atol)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)
