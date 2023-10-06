"""
Test functionality for gval/compare.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"
from numbers import Number

import numpy as np
import pandas as pd
from pytest_cases import parametrize

from tests.conftest import _load_xarray
from gval.comparison.pairing_functions import szudzik_pair_signed, cantor_pair_signed
from gval.utils.schemas import Crosstab_df


numbers_success = [
    0,
    2,
    106.0,
    27,
    13,
    196,
    1e10,
    1.0,
    50.0,
    np.nan,
    np.float64(np.nan),
    -36.89,
    1.9080928438e2,
    5.096279894,
]

numbers_success_expected = [
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    True,
]


@parametrize("number, expected", list(zip(numbers_success, numbers_success_expected)))
def case_is_not_natural_number_successes(number, expected):
    return number, expected


numbers_fail = [
    -1,
    -13,
    -10,
    -10.39023,
    98.80480,
    -1.2e2,
    -1.0,
    48.1,
    np.inf,
    1e99,
    1e25,
]


@parametrize("number", numbers_fail)
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
    (np.float32(-1), np.float64(0), 2),
    (1, -13, 627),
    (-6, 0, 132),
    (6, -130, 67093),
    (np.nan, -130, np.nan),
]


@parametrize("c, b, a", szudzik_pairs_signed)
def case_szudzik_pair_signed(c, b, a):
    return c, b, a


dict_with_nans_data = [
    ({(np.nan, 1): 1, (2, np.float32(np.nan)): 2, (np.float64(np.nan), 5): 3})
]

expected_dict_data = [
    (
        {
            (np.finfo(float).max, 1): 1,
            (2, np.finfo(float).max): 2,
            (np.finfo(float).max, 5): 3,
        }
    )
]


@parametrize(
    "dict_with_nans, expected", list(zip(dict_with_nans_data, expected_dict_data))
)
def case_replace_nans_in_pairing_dict(dict_with_nans, expected):
    return dict_with_nans, expected


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
        [np.float32(np.nan), 2, 3],
        [2, 3, np.float64(np.nan)],
        {
            (float(np.nan), 2): 0,
            (np.nan, 3): 1,
            (np.nan, np.float64(np.nan)): 2,
            (2, 2): 3,
            (2, 3): 4,
            (2, np.float32(np.nan)): 5,
            (3, 2): 6,
            (3, 3): 7,
            (3, np.nan): 8,
        },
    ),
    (
        np.array([5, 6, np.float64(np.nan)]),
        np.array([8, 9]),
        {
            (5, 8): 0,
            (5, 9): 1,
            (6, 8): 2,
            (6, 9): 3,
            (np.float64(np.nan), 8): 4,
            (np.nan, 9): 5,
        },
    ),
    (
        pd.Series([1, 2, np.float32(np.nan)]),
        pd.Series([3, 4, np.nan]),
        {
            (1.0, 3.0): 0,
            (1.0, 4.0): 1,
            (1.0, np.float64(np.nan)): 2,
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


difference_inputs = [
    (1, 3, -2),
    (10.0, 4.0, 6.0),
    (np.nan, 10, np.nan),
    (np.nan, np.nan, np.nan),
]


@parametrize("c, b, expected_value", difference_inputs)
def case_difference(c, b, expected_value):
    return c, b, expected_value


crosstab_dfs = [
    (
        pd.DataFrame({"zone": [0, 1, 2], 0: [10, 100, 200], 1: [50, 25, 100]}),
        "band",
        1,
        Crosstab_df.validate(
            pd.DataFrame(
                {
                    "band": [1, 1, 1, 1, 1, 1],
                    "candidate_values": [0, 1, 2, 0, 1, 2],
                    "benchmark_values": [0, 0, 0, 1, 1, 1],
                    "counts": [10, 100, 200, 50, 25, 100],
                }
            )
        ),
    )
]

crosstab_2d_DataArrayss = [
    (
        _load_xarray(
            "agreement_map_0_aligned_to_candidate_map_0.tif",
            masked=True,
            mask_and_scale=True,
        ).sel(band=1, drop=True),
        Crosstab_df.validate(
            pd.DataFrame(
                {
                    "band": ["1", "1"],
                    "candidate_values": [1.0, 2.0],
                    "benchmark_values": [0.0, 0.0],
                    "agreement_values": [6.0, 20.0],
                    "counts": [10982559.0, 544467.0],
                }
            )
        ),
    ),
    (
        _load_xarray(
            "agreement_map_1_aligned_to_candidate_map_1.tif",
            masked=False,
            mask_and_scale=False,
        ).sel(band=1, drop=True),
        Crosstab_df.validate(
            pd.DataFrame(
                {
                    "band": ["1", "1", "1", "1", "1", "1", "1", "1", "1"],
                    "candidate_values": [
                        1.0,
                        1.0,
                        1.0,
                        2.0,
                        2.0,
                        2.0,
                        -9999.0,
                        -9999.0,
                        -9999.0,
                    ],
                    "benchmark_values": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                    "agreement_values": [
                        6.0,
                        8.0,
                        18.0,
                        20.0,
                        22.0,
                        24.0,
                        399900006.0,
                        399900008.0,
                        399900010.0,
                    ],
                    "counts": [
                        856376.0,
                        199644.0,
                        377453.0,
                        4664.0,
                        231918.0,
                        80368.0,
                        963789.0,
                        113119.0,
                        1086601.0,
                    ],
                }
            )
        ),
    ),
]


@parametrize("agreement_map, expected_df", crosstab_2d_DataArrayss)
def case_crosstab_2d_DataArrays(agreement_map, expected_df):
    return agreement_map, expected_df


crosstab_3d_DataArrayss = [
    (
        _load_xarray(
            "agreement_categorical_multiband_0.tif",
            masked=True,
            mask_and_scale=True,
        ),
        Crosstab_df.validate(
            pd.DataFrame(
                {
                    "band": ["1", "1", "1", "2", "2", "2"],
                    "candidate_values": [1.0, 2.0, -10000.0, 1.0, 2.0, -10000.0],
                    "benchmark_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "agreement_values": [
                        6.0,
                        20.0,
                        399980000.0,
                        6.0,
                        20.0,
                        399980000.0,
                    ],
                    "counts": [
                        10982559.0,
                        544467.0,
                        9489603.0,
                        4845100.0,
                        4845025.0,
                        2470284.0,
                    ],
                }
            )
        ),
    ),
    (
        _load_xarray(
            "agreement_categorical_multiband_1.tif",
            masked=False,
            mask_and_scale=False,
        ),
        Crosstab_df.validate(
            pd.DataFrame(
                {
                    "band": [
                        "1",
                        "1",
                        "1",
                        "1",
                        "1",
                        "1",
                        "1",
                        "1",
                        "2",
                        "2",
                        "2",
                        "2",
                        "2",
                        "2",
                        "2",
                        "2",
                    ],
                    "candidate_values": [
                        1,
                        1,
                        2,
                        2,
                        -9999,
                        -9999,
                        -10000,
                        -10000,
                        1,
                        1,
                        2,
                        2,
                        -9999,
                        -9999,
                        -10000,
                        -10000,
                    ],
                    "benchmark_values": [
                        0,
                        2,
                        0,
                        2,
                        0,
                        2,
                        0,
                        2,
                        0,
                        2,
                        0,
                        2,
                        0,
                        2,
                        0,
                        2,
                    ],
                    "agreement_values": [
                        6.0,
                        18.0,
                        20.0,
                        24.0,
                        399900006.0,
                        399900010.0,
                        399980000.0,
                        399980004.0,
                        6.0,
                        18.0,
                        20.0,
                        24.0,
                        399900006.0,
                        399900010.0,
                        399980000.0,
                        399980004.0,
                    ],
                    "counts": [
                        10982559.0,
                        679166.0,
                        544467.0,
                        2624332.0,
                        4740048.0,
                        2050.0,
                        9489603.0,
                        4415.0,
                        4845100.0,
                        4844092.0,
                        4845025.0,
                        4847966.0,
                        2370734.0,
                        2371364.0,
                        2470284.0,
                        2472075.0,
                    ],
                }
            )
        ),
    ),
]


@parametrize("agreement_map, expected_df", crosstab_3d_DataArrayss)
def case_crosstab_3d_DataArrays(agreement_map, expected_df):
    return agreement_map, expected_df


@parametrize(
    "agreement_map, expected_df",
    crosstab_2d_DataArrayss + crosstab_3d_DataArrayss,
)
def case_crosstab_DataArrays_success(agreement_map, expected_df):
    return agreement_map, expected_df


crosstab_DataArrays_fails = [
    _load_xarray(
        "agreement_categorical_multiband_0.tif",
        masked=True,
        mask_and_scale=True,
    ).expand_dims({"dummy_dim": 1}),
]


@parametrize("agreement_map", crosstab_DataArrays_fails)
def case_crosstab_DataArrays_fail(agreement_map):
    return agreement_map


_crosstab_Datasets = crosstab_3d_DataArrayss
_input_datasets = [
    (
        _load_xarray(
            "agreement_categorical_multiband_0.tif",
            masked=True,
            mask_and_scale=True,
            band_as_variable=True,
        ),
    ),
    (
        _load_xarray(
            "agreement_categorical_multiband_1.tif",
            masked=False,
            mask_and_scale=False,
            band_as_variable=True,
        ),
    ),
]

expected_dfs = [(c[1],) for c in crosstab_3d_DataArrayss]


@parametrize(
    "agreement_map, expected_df",
    [i + ii for i, ii in zip(_input_datasets, expected_dfs)],
)
def case_crosstab_Datasets(agreement_map, expected_df):
    return agreement_map, expected_df


compute_agreement_maps_success = [
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_szudzik_aligned_to_candidate_map_0.tif"),
        szudzik_pair_signed,
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_cantor_aligned_to_candidate_map_0.tif"),
        cantor_pair_signed,
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_pairing_aligned_to_candidate_map_0.tif"),
        "pairing_dict",
        None,
        [-9999, 1, 2],
        [0, 2],
        None,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_pairing_aligned_to_candidate_map_0.tif"),
        "pairing_dict",
        {(-9999, 0): 0, (-9999, 2): 1, (1, 0): 2, (1, 2): 3, (2, 0): 4, (2, 2): 5},
        None,
        None,
        None,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_szudzik_aligned_to_candidate_map_0_nodata.tif"),
        szudzik_pair_signed,
        None,
        None,
        None,
        399900006,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray(
            "agreement_map_00_szudzik_aligned_to_candidate_map_0_nodata.tif",
            masked=True,
            mask_and_scale=True,
        ),
        szudzik_pair_signed,
        None,
        None,
        None,
        399900006,
        True,
    ),
]


@parametrize(
    "candidate_map, benchmark_map, agreement_map, comparison_function, pairing_dict, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata",
    compute_agreement_maps_success,
)
def case_compute_agreement_map_success(
    candidate_map,
    benchmark_map,
    agreement_map,
    comparison_function,
    pairing_dict,
    allow_candidate_values,
    allow_benchmark_values,
    nodata,
    encode_nodata,
):
    return (
        _load_xarray(candidate_map),
        _load_xarray(benchmark_map),
        agreement_map,
        comparison_function,
        pairing_dict,
        allow_candidate_values,
        allow_benchmark_values,
        nodata,
        encode_nodata,
    )


compute_agreement_maps_fail = [
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        "pairing_dict",
        None,
        None,
        None,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        szudzik_pair_signed,
        None,
        None,
        None,
        True,
    ),
]


@parametrize(
    "candidate_map, benchmark_map, comparison_function, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata",
    compute_agreement_maps_fail,
)
def case_compute_agreement_map_fail(
    candidate_map,
    benchmark_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
    nodata,
    encode_nodata,
):
    return (
        _load_xarray(candidate_map),
        _load_xarray(benchmark_map),
        comparison_function,
        allow_candidate_values,
        allow_benchmark_values,
        nodata,
        encode_nodata,
    )


comparison_processing_agreement_maps_success = [
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_szudzik_aligned_to_candidate_map_0.tif"),
        "szudzik",
        None,
        None,
        None,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_cantor_aligned_to_candidate_map_0.tif"),
        "cantor",
        None,
        None,
        None,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_pairing_aligned_to_candidate_map_0.tif"),
        "pairing_dict",
        [-9999, 1, 2],
        [0, 2],
        None,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray("agreement_map_00_szudzik_aligned_to_candidate_map_0_nodata.tif"),
        "szudzik",
        None,
        None,
        399900006,
        None,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        _load_xarray(
            "agreement_map_00_szudzik_aligned_to_candidate_map_0_nodata.tif",
            masked=True,
            mask_and_scale=True,
        ),
        "szudzik",
        None,
        None,
        399900006,
        True,
    ),
]


@parametrize(
    "candidate_map, benchmark_map, agreement_map, comparison_function, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata",
    comparison_processing_agreement_maps_success,
)
def case_comparison_processing_agreement_maps_success(
    candidate_map,
    benchmark_map,
    agreement_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
    nodata,
    encode_nodata,
):
    return (
        _load_xarray(candidate_map),
        _load_xarray(benchmark_map),
        agreement_map,
        comparison_function,
        allow_candidate_values,
        allow_benchmark_values,
        nodata,
        encode_nodata,
    )


comparison_processing_agreement_maps_fail = [
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        "pairing_dict",
        None,
        None,
        None,
        None,
        ValueError,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        "szudzik",
        None,
        None,
        None,
        True,
        ValueError,
    ),
    (
        "candidate_map_0_aligned_to_candidate_map_0.tif",
        "benchmark_map_0_aligned_to_candidate_map_0.tif",
        "arb",
        None,
        None,
        None,
        True,
        KeyError,
    ),
]


@parametrize(
    "candidate_map, benchmark_map, comparison_function, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata, exception",
    comparison_processing_agreement_maps_fail,
)
def case_comparison_processing_agreement_maps_fail(
    candidate_map,
    benchmark_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
    nodata,
    encode_nodata,
    exception,
):
    return (
        _load_xarray(candidate_map),
        _load_xarray(benchmark_map),
        comparison_function,
        allow_candidate_values,
        allow_benchmark_values,
        nodata,
        encode_nodata,
        exception,
    )


comparison_args = [
    {"name": "test_func"},
    {"name": "test_func2", "vectorize_func": True},
]


def pass1(c: float, b: int) -> Number:
    return c + b


def pass2(c: float, b: float) -> Number:
    return c + b


comparison_funcs = [pass1, pass2]


@parametrize("args, func", list(zip(comparison_args, comparison_funcs)))
def case_comparison_register_function(args, func):
    return args, func


comparison_args = [
    {"name": "szudzik"},
    {"name": "test2"},
    {"name": "test2"},
    {"name": "test2"},
]


def fail1(c: float, b: int) -> Number:
    return c + b


def fail2(c: str, b: int) -> Number:
    return c + b


def fail3(c: float) -> Number:
    return c


def fail4(c: float, b: float) -> str:
    return c + b


comparison_fail_funcs = [fail1, fail2, fail3, fail4]
exceptions = [KeyError, TypeError, TypeError, TypeError]


@parametrize(
    "args, func, exception",
    list(zip(comparison_args, comparison_fail_funcs, exceptions)),
)
def case_comparison_register_function_fail(args, func, exception):
    return args, func, exception


class Tester:
    @staticmethod
    def pass5(c: int, b: int) -> Number:
        return c + b

    @staticmethod
    def pass6(c: int, b: int) -> Number:
        return c + b


class Tester2:
    @staticmethod
    def pass7(c: int, b: int) -> Number:
        return c + b

    @staticmethod
    def pass8(c: int, b: int) -> Number:
        return c + b


comparison_names = [["pass5", "pass6"], ["pass7", "pass8"]]
comparison_args = [{}, {"vectorize_func": True}]
comparison_class = [Tester, Tester2]


@parametrize(
    "names, args, cls", list(zip(comparison_names, comparison_args, comparison_class))
)
def case_comparison_register_class(names, args, cls):
    return names, args, cls


class TesterFail1:
    @staticmethod
    def fail6(c: str, b: int) -> Number:
        return c + b


class TesterFail2:
    @staticmethod
    def szudzik(c: int, b: int) -> Number:
        return c + b


comparison_args = [{}, {"vectorize_func": True}]
comparison_class = [TesterFail1, TesterFail2]
exceptions = [TypeError, KeyError]


@parametrize(
    "args, cls, exception", list(zip(comparison_args, comparison_class, exceptions))
)
def case_comparison_register_class_fail(args, cls, exception):
    return args, cls, exception


stat_funcs = ["szudzik", "pairing_dict"]
stat_params = [["c", "b"], ["c", "b"]]


@parametrize("name, params", list(zip(stat_funcs, stat_params)))
def case_comparison_get_param(name, params):
    return name, params


stat_funcs = ["arbitrary"]


@parametrize("name", stat_funcs)
def case_comparison_get_param_fail(name):
    return name
