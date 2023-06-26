"""
Test functionality for computing_continuous_metrics.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"


import pandas as pd
from pytest_cases import parametrize_with_cases
from pytest import raises
import numpy as np

from gval import ContStats as con_stat
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
        agreement_map=candidate_map - benchmark_map,
        candidate_map=candidate_map,
        benchmark_map=benchmark_map,
        metrics=metrics,
    )

    pd.testing.assert_frame_equal(
        metrics_df, expected_df, check_dtype=False
    ), "Compute statistics did not return expected values"


@parametrize_with_cases(
    "names, error_map, exception", glob="compute_continuous_statistics_fail"
)
def test_compute_continuous_statistics_fail(names, error_map, exception):
    """tests compute statistics fail function"""

    args = {
        "agreement_map": error_map,
        "metrics": names,
        "candidate_map": None,
        "benchmark_map": None,
    }

    with np.errstate(divide="ignore"):
        with raises(exception):
            # NOTE: Removed bc this should be handled within process_statistics.
            # stat_names = cat_stat.available_functions() if names == "all" else names
            _compute_continuous_metrics(**args)


@parametrize_with_cases("args, func", glob="register_continuous_function")
def test_register_continuous_function(args, func):
    """tests register func function"""

    con_stat.register_function(**args)(func)


@parametrize_with_cases(
    "args, func, exception", glob="register_continuous_function_fail"
)
def test_register_continuous_function_fail(args, func, exception):
    """tests register func fail function"""

    with raises(exception):
        con_stat.register_function(**args)(func)


@parametrize_with_cases("names, args, cls", glob="register_class_continuous")
def test_register_class_continuous(names, args, cls):
    """tests register class continuous function"""

    con_stat.register_function_class(**args)(cls)

    if [name in con_stat.registered_functions for name in names] != [True] * len(names):
        assert False, "Unable to register all class functions"


@parametrize_with_cases("args, cls, exception", glob="register_class_continuous_fail")
def test_register_class_continuous_fail(args, cls, exception):
    """tests register class continuous fail function"""

    with raises(exception):
        con_stat.register_function_class(**args)(cls)


@parametrize_with_cases("name, params", glob="get_param_continuous")
def test_get_param_continuous(name, params):
    """tests get param continuous function"""

    _params = con_stat.get_parameters(name)
    assert _params == params


@parametrize_with_cases("name", glob="get_param_continuous_fail")
def test_get_param_continuous_fail(name):
    """tests get param continuous fail function"""

    with raises(KeyError):
        _ = con_stat.get_parameters(name)


def test_get_all_param_continuous():
    """tests get all params function"""

    try:
        con_stat.get_all_parameters()
    except KeyError:
        assert False, "Signature dict not present or keys changed"


def test_available_functions_continuous():
    """tests get available functions"""

    a_funcs = con_stat.available_functions()

    assert isinstance(a_funcs, list)
