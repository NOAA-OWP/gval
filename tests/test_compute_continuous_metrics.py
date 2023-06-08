"""
Test functionality for computing_continuous_metrics.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"


import pandas as pd
from pytest_cases import parametrize_with_cases

from gval.comparison.compute_continuous_metrics import _compute_continuous_metrics


@parametrize_with_cases(
    "candidate_map, benchmark_map, metrics, expected_df",
    glob="compute_continuous_metrics_success",
)
def test_compute_continuous_metrics_success(
    candidate_map, benchmark_map, metrics, expected_df
):
    """tests continuous metrics functions"""

    # compute continuous metrics
    metrics_df = _compute_continuous_metrics(
        candidate_map=candidate_map, benchmark_map=benchmark_map, metrics=metrics
    )

    pd.testing.assert_frame_equal(
        metrics_df, expected_df, check_dtype=False
    ), "Compute statistics did not return expected values"
