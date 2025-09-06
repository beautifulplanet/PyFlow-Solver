import numpy as np
import os
import pytest

@pytest.fixture(autouse=True)
def global_seed():
    seed = int(os.environ.get('TEST_GLOBAL_SEED','12345'))
    np.random.seed(seed)
    yield
