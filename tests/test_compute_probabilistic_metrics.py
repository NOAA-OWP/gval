"""
Test functionality for compute_probabilistic_metrics.py
"""

__author__ = "Fernando Aristizabal"

import xarray as xr
import pandas as pd
from pytest_cases import parametrize_with_cases
from pytest import raises, warns

from tests.conftest import _compare_metrics_df_with_xarray
from gval.comparison.compute_probabilistic_metrics import _compute_probabilistic_metrics


'''FAILED tests/test_continuous_metrics.py::test_range_normalized_root_mean_squared_error[range_normalized_root_mean_squared_error-candidate_continuous_1.tif-benchmark_continuous_1.tif-load_options1-expected_value1] - AssertionError: 
FAILED tests/test_continuous_metrics.py::test_range_normalized_mean_absolute_error[range_normalized_mean_absolute_error-candidate_continuous_1.tif-benchmark_continuous_1.tif-load_options1-expected_value1] - AssertionError: 
'''

@parametrize_with_cases(
    "candidate_map, benchmark_map, compute_kwargs, expected_df",
    glob="compute_prob_metrics_*_success",
)
def test_compute_probabilistic_metrics(
    candidate_map, benchmark_map, compute_kwargs, expected_df
):
    """tests probabilistic metrics computation"""

    # compute categorical metrics
    metrics_df = _compute_probabilistic_metrics(
        candidate_map=candidate_map,
        benchmark_map=benchmark_map,
        **compute_kwargs
    )

    _compare_metrics_df_with_xarray(metrics_df, expected_df)


@parametrize_with_cases(
    "candidate_map, benchmark_map, compute_kwargs, expected_error",
    glob="compute_prob_metrics_*_fail",
)
def test_compute_probabilistic_metrics_fail(
    candidate_map, benchmark_map, compute_kwargs, expected_error
):
    """tests probabilistic metrics computation"""

    # compute categorical metrics
    with raises(expected_error):
        _compute_probabilistic_metrics(
            candidate_map=candidate_map,
            benchmark_map=benchmark_map,
            **compute_kwargs
        )

@parametrize_with_cases(
    "candidate_map, benchmark_map, compute_kwargs, expected_warning",
    glob="compute_prob_metrics_*_warns",
)
def test_compute_probabilistic_metrics_warns(
    candidate_map, benchmark_map, compute_kwargs, expected_warning
):
    """tests probabilistic metrics computation warnings"""

    # compute categorical metrics
    with warns(expected_warning):
        _compute_probabilistic_metrics(
            candidate_map=candidate_map,
            benchmark_map=benchmark_map,
            **compute_kwargs
        )
