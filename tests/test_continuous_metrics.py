"""
Test functionality for continuous value metrics.
"""

# __all__ = ['*']

from functools import wraps

import numpy as np
from pytest_cases import parametrize_with_cases

from tests.conftest import _assert_pairing_dict_equal
from gval.statistics import continuous_stat_funcs


def assert_logic_decorator(test_function):
    @wraps(test_function)
    def wrapper(candidate_map, benchmark_map, expected_value):
        # Call the test function
        result = test_function(candidate_map, benchmark_map, expected_value)

        # Implement the assert logic
        if isinstance(expected_value, dict):
            _assert_pairing_dict_equal(result, expected_value)
        else:
            np.testing.assert_almost_equal(result, expected_value)

    return wrapper


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="mean_absolute_error",
)
@assert_logic_decorator
def test_mean_absolute_error(candidate_map, benchmark_map, expected_value):
    return continuous_stat_funcs.mean_absolute_error(candidate_map - benchmark_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="mean_squared_error",
)
@assert_logic_decorator
def test_mean_squared_error(candidate_map, benchmark_map, expected_value):
    return continuous_stat_funcs.mean_squared_error(candidate_map - benchmark_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="root_mean_squared_error",
)
@assert_logic_decorator
def test_root_mean_squared_error(candidate_map, benchmark_map, expected_value):
    return continuous_stat_funcs.root_mean_squared_error(candidate_map - benchmark_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="mean_signed_error",
)
@assert_logic_decorator
def test_mean_signed_error(candidate_map, benchmark_map, expected_value):
    return continuous_stat_funcs.mean_signed_error(candidate_map - benchmark_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="mean_percentage_error",
)
@assert_logic_decorator
def test_mean_percentage_error(candidate_map, benchmark_map, expected_value):
    return continuous_stat_funcs.mean_percentage_error(
        candidate_map - benchmark_map, benchmark_map
    )


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="mean_absolute_percentage_error",
)
@assert_logic_decorator
def test_mean_absolute_percentage_error(candidate_map, benchmark_map, expected_value):
    return continuous_stat_funcs.mean_absolute_percentage_error(
        candidate_map - benchmark_map, benchmark_map
    )


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="mean_normalized_root_mean_squared_error",
)
@assert_logic_decorator
def test_mean_normalized_root_mean_squared_error(
    candidate_map, benchmark_map, expected_value
):
    return continuous_stat_funcs.mean_normalized_root_mean_squared_error(
        candidate_map - benchmark_map, benchmark_map
    )


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="range_normalized_root_mean_squared_error",
)
@assert_logic_decorator
def test_range_normalized_root_mean_squared_error(
    candidate_map, benchmark_map, expected_value
):
    return continuous_stat_funcs.range_normalized_root_mean_squared_error(
        candidate_map - benchmark_map, benchmark_map
    )


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="mean_normalized_mean_absolute_error",
)
@assert_logic_decorator
def test_mean_normalized_mean_absolute_error(
    candidate_map, benchmark_map, expected_value
):
    return continuous_stat_funcs.mean_normalized_mean_absolute_error(
        candidate_map - benchmark_map, benchmark_map
    )


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="range_normalized_mean_absolute_error",
)
@assert_logic_decorator
def test_range_normalized_mean_absolute_error(
    candidate_map, benchmark_map, expected_value
):
    return continuous_stat_funcs.range_normalized_mean_absolute_error(
        candidate_map - benchmark_map, benchmark_map
    )


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="coefficient_of_determination",
)
@assert_logic_decorator
def test_coefficient_of_determination(candidate_map, benchmark_map, expected_value):
    return continuous_stat_funcs.coefficient_of_determination(
        candidate_map - benchmark_map, benchmark_map
    )


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_value",
    glob="symmetric_mean_absolute_percentage_error",
)
@assert_logic_decorator
def test_symmetric_mean_absolute_percentage_error(
    candidate_map, benchmark_map, expected_value
):
    return continuous_stat_funcs.symmetric_mean_absolute_percentage_error(
        candidate_map - benchmark_map, candidate_map, benchmark_map
    )
