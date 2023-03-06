"""
Test functionality for gval/compare.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np
import pandas as pd
from pytest_cases import parametrize

from tests.conftest import _load_xarray
from gval.compare import szudzik_pair_signed, cantor_pair_signed


numbers = [0, 2, 106.0, 27, 13, 196, 1e10, 1.0, 50.0, np.nan, np.float64(np.nan)]


@parametrize("number", numbers)
def case_is_not_natural_number_successes(number):
    return number


numbers = [-1, -13, -10, -10.39023, 98.80480, -1.2e2, -1.0, 48.1, np.inf, 1e99, 1e25]


@parametrize("number", numbers)
def case_is_not_natural_number_failures(number):
    return number


cantor_pairs = [(1, 0, 1), (1, 13, 118), (6, 0, 21), (6, 13, 203)]


@parametrize("c, b, a", cantor_pairs)
def case_cantor_pair(c, b, a):
    return c, b, a


cantor_pairs_signed = [
    (-1, 0, 1),
    (1, -13, 403),
    (-6, 0, 66),
    (6, -130, 37115),
    (np.nan, -130, np.nan),
]


@parametrize("c, b, a", cantor_pairs_signed)
def case_cantor_pair_signed(c, b, a):
    return c, b, a


szudzik_pairs = [(1, 0, 2), (1, 13, 170), (6, 0, 42), (6, 13, 175)]


@parametrize("c, b, a", szudzik_pairs)
def case_szudzik_pair(c, b, a):
    return c, b, a


szudzik_pairs_signed = [
    (-1, 0, 2),
    (1, -13, 627),
    (-6, 0, 132),
    (6, -130, 67093),
    (np.nan, -130, np.nan),
]


@parametrize("c, b, a", szudzik_pairs_signed)
def case_szudzik_pair_signed(c, b, a):
    return c, b, a


pairing_dicts = [
    (
        range(3),
        range(3, 5),
        {(0, 3): 0, (0, 4): 1, (1, 3): 2, (1, 4): 3, (2, 3): 4, (2, 4): 5},
    ),
    (
        [10, 11, 12],
        [1, 5, 8],
        {
            (10, 1): 0,
            (10, 5): 1,
            (10, 8): 2,
            (11, 1): 3,
            (11, 5): 4,
            (11, 8): 5,
            (12, 1): 6,
            (12, 5): 7,
            (12, 8): 8,
        },
    ),
    (
        [np.nan, 2, 3],
        [2, 3, np.nan],
        {
            (np.nan, 2): 0,
            (np.nan, 3): 1,
            (np.nan, np.nan): 2,
            (2, 2): 3,
            (2, 3): 4,
            (2, np.nan): 5,
            (3, 2): 6,
            (3, 3): 7,
            (3, np.nan): 8,
        },
    ),
    (
        np.array([5, 6, np.nan]),
        np.array([8, 9]),
        {
            (5, 8): 0,
            (5, 9): 1,
            (6, 8): 2,
            (6, 9): 3,
            (np.nan, 8): 4,
            (np.nan, 9): 5,
        },
    ),
    (
        pd.Series([1, 2, np.nan]),
        pd.Series([3, 4, np.nan]),
        {
            (1.0, 3.0): 0,
            (1.0, 4.0): 1,
            (1.0, np.nan): 2,
            (2.0, 3.0): 3,
            (2.0, 4.0): 4,
            (2.0, np.nan): 5,
            (np.nan, 3.0): 6,
            (np.nan, 4.0): 7,
            (np.nan, np.nan): 8,
        },
    ),
]


@parametrize(
    "unique_candidate_values, unique_benchmark_values, expected_dict", pairing_dicts
)
def case_make_pairing_dict(
    unique_candidate_values, unique_benchmark_values, expected_dict
):
    return unique_candidate_values, unique_benchmark_values, expected_dict


pairing_dict_fn_inputs = [
    (1, 2, {(1, 2): 3}, 3),
    (9, 10, {(9, 10.0): 1}, 1),
    (-1, 10, {(-1, 10): np.nan}, np.nan),
]


@parametrize("c, b, pairing_dict, expected_value", pairing_dict_fn_inputs)
def case_pairing_dict_fn(c, b, pairing_dict, expected_value):
    return c, b, pairing_dict, expected_value


crosstab_xarrays = [
    (
        "candidate_map_0.tif",
        "benchmark_map_0.tif",
        pd.DataFrame(
            {
                "zone": [-9999, 1, 2],
                0: [60420648, 41687643, 2232777],
                2: [143467, 2163307, 10641990],
            }
        ),
    )
]


@parametrize("candidate_map, benchmark_map, expected_df", crosstab_xarrays)
def case_crosstab_xarray(candidate_map, benchmark_map, expected_df):
    return (_load_xarray(candidate_map), _load_xarray(benchmark_map), expected_df)


compute_agreement_xarrays = [
    (
        "candidate_map_0.tif",
        "benchmark_map_0.tif",
        "agreement_map_szudzik.tif",
        szudzik_pair_signed,
        None,
        None,
    ),
    (
        "candidate_map_0.tif",
        "benchmark_map_0.tif",
        "agreement_map_cantor.tif",
        cantor_pair_signed,
        None,
        None,
    ),
    (
        "candidate_map_0.tif",
        "benchmark_map_0.tif",
        "agreement_map_pairing_dict.tif",
        "pairing_dict",
        [-9999, 1, 2],
        [0, 2],
    ),
]


@parametrize(
    "candidate_map, benchmark_map, agreement_map, comparison_function, allow_candidate_values, allow_benchmark_values",
    compute_agreement_xarrays,
)
def case_compute_agreement_xarray(
    candidate_map,
    benchmark_map,
    agreement_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
):
    return (
        _load_xarray(candidate_map),
        _load_xarray(benchmark_map),
        _load_xarray(agreement_map),
        comparison_function,
        allow_candidate_values,
        allow_benchmark_values,
    )
