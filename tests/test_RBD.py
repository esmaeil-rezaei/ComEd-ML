import numpy as np
import pytest
from src.utils import RBD_method

def test_RBD_method_basic():

    # Create a small random matrix (5x5)
    np.random.seed(42)
    data = np.random.rand(5, 5)
    col = 3
    
    bases = RBD_method(data, col)
    
    assert bases.shape == (data.shape[0], col), "Output shape mismatch"
    
    for i in range(bases.shape[1]):
        norm = np.linalg.norm(bases[:, i])
        assert np.isclose(norm, 1.0, atol=1e-5), f"Column {i} is not normalized"

def test_RBD_method_single_column():
    data = np.array([[1, 2], [3, 4]], dtype=float)
    col = 1
    bases = RBD_method(data, col)
    
    # Check shape
    assert bases.shape == (2, 1)
    
    # Check normalization
    assert np.isclose(np.linalg.norm(bases[:, 0]), 1.0)
