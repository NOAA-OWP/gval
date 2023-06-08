"""
Test functionality for computing_continuous_metrics.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest_cases import parametrize
import pandas as pd

from tests.conftest import _load_xarray


expected_dfs = [
    pd.DataFrame(
        {
            "band": {0: "1"},
            "mean_absolute_error": 0.3173885941505432,
            "mean_squared_error": 0.30277130007743835,
            "root_mean_squared_error": 0.5502465963363647,
            "mean_percentage_error": 0.015025136061012745,
            "mean_absolute_percentage_error": 0.15956786274909973,
            "coefficient_of_determination": -0.06615996360778809,
            "normalized_mean_absolute_error": 0.16163060069084167,
            "normalized_root_mean_squared_error": 0.27127230167388916,
        }
    ),
    pd.DataFrame(
        {
            "band": ["1", "2"],
            "mean_absolute_error": [0.48503121733665466, 0.48503121733665466],
            "mean_squared_error": [0.5341722369194031, 0.5341722369194031],
            "root_mean_squared_error": [0.7308709025382996, 0.7308709025382996],
        }
    ),
]

candidate_maps = ["candidate_continuous_0.tif", "candidate_continuous_1.tif"]
benchmark_maps = ["benchmark_continuous_0.tif", "benchmark_continuous_1.tif"]

all_load_options = [
    {"mask_and_scale": True},
    {"mask_and_scale": True, "band_as_variable": True},
]

metrics_input = [
    "all",
    ["mean_absolute_error", "mean_squared_error", "root_mean_squared_error"],
]


@parametrize(
    "candidate_map, benchmark_map, load_options, metrics, expected_df",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            all_load_options,
            metrics_input,
            expected_dfs,
        )
    ),
)
def case_compute_continuous_metrics_success(
    candidate_map, benchmark_map, load_options, metrics, expected_df
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        metrics,
        expected_df,
    )
