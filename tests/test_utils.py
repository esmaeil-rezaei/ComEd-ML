import os
import numpy as np
import pandas as pd
import tempfile
from src.utils import (
    save_npy, load_npy,
    estimate_mean_absolute_percentage, estimate_coefficient_of_error, relative_error,
    PCA_method, make_data_r_time_intervals_dependent
)

def test_save_and_load_npy():
    arr = np.array([[1, 2], [3, 4]])
    with tempfile.TemporaryDirectory() as tmpdir:
        save_npy("test_array", tmpdir, arr)
        loaded = load_npy("test_array.npy", tmpdir)
        assert np.array_equal(arr, loaded)

def test_error_metrics():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    mape = estimate_mean_absolute_percentage(y_true, y_pred)
    assert 0 <= mape <= 1

    cv = estimate_coefficient_of_error(y_true, y_pred)
    assert cv > 0

    rel = relative_error(y_true, y_pred)
    assert len(rel) == len(y_true)

def test_pca_method_reduces_dim():
    data = np.random.rand(100, 10)
    reduced, pca = PCA_method(data, dim_reduction_size=3)
    assert reduced.shape[1] == 3

def test_make_data_r_time_intervals_dependent():
    data = np.random.rand(10, 2)  # 10 time steps, 2 users
    result = make_data_r_time_intervals_dependent(data, r=2)
    # Shape: (m-r, r*n+1) => (8, 2*2) = (8, 4)
    assert result.shape == (8, 4)
