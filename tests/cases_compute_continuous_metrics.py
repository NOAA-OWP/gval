"""
Test functionality for computing_continuous_metrics.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from numbers import Number
from typing import Union

import numpy as np
from pytest_cases import parametrize
import pandas as pd
import xarray as xr
from tests.conftest import _load_xarray


expected_dfs = [
    pd.DataFrame(
        {
            "band": {0: "1"},
            "coefficient_of_determination": -0.06615996360778809,
            "mean_absolute_error": 0.3173885941505432,
            "mean_absolute_percentage_error": 0.15956786274909973,
            "mean_normalized_mean_absolute_error": 0.16163060069084167,
            "mean_normalized_root_mean_squared_error": 0.2802138924598694,
            "mean_percentage_error": -0.015025136061012745,
            "mean_signed_error": -0.02950434572994709,
            "mean_squared_error": 0.30277130007743835,
            "range_normalized_mean_absolute_error": 0.27127230167388916,
            "range_normalized_root_mean_squared_error": 0.4702962636947632,
            "root_mean_squared_error": 0.5502465963363647,
            "symmetric_mean_absolute_percentage_error": 0.1628540771054295,
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
candidate_maps_fail = [
    "candidate_continuous_0_fail.tif",
    "candidate_continuous_0_fail.tif",
    "candidate_continuous_0_fail.tif",
    None,
]

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


stat_names = ["all", "all", "non_existent_function", "all"]
exceptions = [ValueError, ValueError, KeyError, ValueError]


@parametrize(
    "names, error_map, exception",
    list(zip(stat_names, candidate_maps_fail, exceptions)),
)
def case_compute_continuous_statistics_fail(names, error_map, exception):
    test_map = _load_xarray(error_map) if error_map is not None else error_map
    return names, test_map, exception


stat_args = [{"name": "test_func"}, {"name": "test_func2"}]


def pass1(error: Union[xr.DataArray, xr.Dataset]) -> Number:
    return (error + 1).mean()


def pass2(
    error: Union[xr.Dataset, xr.DataArray],
    benchmark_map: Union[xr.Dataset, xr.DataArray],
) -> Number:
    return ((error + benchmark_map) / 1000).sum()


stat_funcs = [pass1, pass2]


@parametrize("args, func", list(zip(stat_args, stat_funcs)))
def case_register_continuous_function(args, func):
    return args, func


stat_args = [
    {"name": "mean_percentage_error"},
    {"name": "test2"},
    {"name": "test2"},
    {"name": "test2"},
    {"name": "test2"},
]


def fail1(error: Union[xr.Dataset, xr.DataArray]) -> Number:
    return error.mean()


def fail2(arb: Union[xr.Dataset, xr.DataArray]) -> Number:
    return arb.mean()


def fail3(error: np.array) -> Number:
    return error.mean()


def fail4(error: Union[xr.Dataset, xr.DataArray]) -> str:
    return error.mean()


def fail5() -> Number:
    return 8.0


stat_fail_funcs = [fail1, fail2, fail3, fail4, fail5]
exceptions = [KeyError, TypeError, TypeError, TypeError, TypeError]


@parametrize("args, func, exception", list(zip(stat_args, stat_fail_funcs, exceptions)))
def case_register_continuous_function_fail(args, func, exception):
    return args, func, exception


class Tester:
    @staticmethod
    def pass5(
        error: Union[xr.Dataset, xr.DataArray],
        benchmark_map: Union[xr.Dataset, xr.DataArray],
    ) -> Number:
        return error / benchmark_map

    @staticmethod
    def pass6(error: Union[xr.Dataset, xr.DataArray]) -> Number:
        return error * 1.01


stat_names = [["pass5", "pass6"]]
stat_args = [{}]
stat_class = [Tester]


@parametrize("names, args, cls", list(zip(stat_names, stat_args, stat_class)))
def case_register_class_continuous(names, args, cls):
    return names, args, cls


class TesterFail1:
    @staticmethod
    def fail6(rp: int, fn: int) -> float:
        return rp + fn


class TesterFail2:
    @staticmethod
    def mean_absolute_error(error: Union[xr.DataArray, xr.Dataset]) -> Number:
        return error + 0.01


stat_args = [{}, {}]
stat_class = [TesterFail1, TesterFail2]
exceptions = [TypeError, KeyError]


@parametrize("args, cls, exception", list(zip(stat_args, stat_class, exceptions)))
def case_register_class_continuous_fail(args, cls, exception):
    return args, cls, exception


stat_funcs = ["mean_absolute_error", "symmetric_mean_absolute_percentage_error"]
stat_params = [["error"], ["error", "candidate_map", "benchmark_map"]]


@parametrize("name, params", list(zip(stat_funcs, stat_params)))
def case_get_param_continuous(name, params):
    return name, params


stat_funcs = ["arbitrary"]


@parametrize("name", stat_funcs)
def case_get_param_continuous_fail(name):
    return name
