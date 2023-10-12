"""
Test functionality for gval/compare.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases
import xarray as xr
from pytest import raises

from gval.comparison.pairing_functions import (
    _is_not_natural_number,
    cantor_pair,
    cantor_pair_signed,
    szudzik_pair,
    szudzik_pair_signed,
    _make_pairing_dict,
    _make_pairing_dict_fn,
    PairingDict,
    difference,
)
from gval.comparison.agreement import _compute_agreement_map
from gval.comparison.tabulation import (
    _crosstab_2d_DataArrays,
    _crosstab_3d_DataArrays,
    _crosstab_DataArrays,
    _crosstab_Datasets,
)
from gval.comparison.compute_comparison import ComparisonProcessing
from gval.utils.schemas import Crosstab_df

from tests.conftest import _assert_pairing_dict_equal


compare_proc = ComparisonProcessing()


@parametrize_with_cases("number, expected", glob="is_not_natural_number_successes")
def test_is_not_natural_number_success(number, expected):
    assert expected == _is_not_natural_number(number, False), "Expected natural number"


@parametrize_with_cases("number", glob="is_not_natural_number_failures")
def test_is_not_natural_number_failures(number):
    with pytest.raises(ValueError):
        _is_not_natural_number(number, True)


@parametrize_with_cases("c, b, a", glob="cantor_pair")
def test_cantor_pair(c, b, a):
    """tests cantor pairing function"""
    np.testing.assert_equal(
        cantor_pair(c, b), a
    ), "Cantor function output does not match expected value"


@parametrize_with_cases("c, b, a", glob="cantor_pair_signed")
def test_cantor_pair_signed(c, b, a):
    """
    tests cantor pairing function
    FIXME: Figure out why RuntimeWarning is occurring for test cases with np.nan.
    """
    np.testing.assert_equal(
        cantor_pair_signed(c, b), a
    ), "Signed cantor function output does not match expected value"


@parametrize_with_cases("c, b, a", glob="szudzik_pair")
def test_szudzik_pair(c, b, a):
    np.testing.assert_equal(
        szudzik_pair(c, b), a
    ), "szudzik function output does not match expected value"


@parametrize_with_cases("c, b, a", glob="szudzik_pair_signed")
def test_szudzik_pair_signed(c, b, a):
    """
    Tests szudzik pairing function
    FIXME: Figure out why RuntimeWarning is occurring for test cases with np.nan.
    """
    np.testing.assert_equal(
        szudzik_pair_signed(c, b), a
    ), "Signed szudzik function output does not match expected value"


@parametrize_with_cases("dict_with_nans, expected", glob="replace_nans_in_pairing_dict")
def test_replace_nans_in_pairing_dict(dict_with_nans, expected):
    _assert_pairing_dict_equal(PairingDict(dict_with_nans), expected)


@parametrize_with_cases(
    "unique_candidate_values, unique_benchmark_values, expected_dict",
    glob="make_pairing_dict",
)
def test_make_pairing_dict(
    unique_candidate_values, unique_benchmark_values, expected_dict
):
    """Tests creation of a pairing dictionary"""

    computed_dict = _make_pairing_dict(unique_candidate_values, unique_benchmark_values)
    expected_dict = PairingDict(expected_dict)

    _assert_pairing_dict_equal(computed_dict, expected_dict)


@parametrize_with_cases("c, b, pairing_dict, expected_value", glob="pairing_dict_fn")
def test_pairing_dict_fn(c, b, pairing_dict, expected_value):
    """Tests pairing dictionary function"""

    pairing_dict_fn = _make_pairing_dict_fn(pairing_dict)
    computed_value = pairing_dict_fn(c, b)

    np.testing.assert_equal(computed_value, expected_value)


@parametrize_with_cases("c, b, expected_value", glob="difference")
def test_difference(c, b, expected_value):
    """Tests difference comparison function"""

    np.testing.assert_equal(difference(c, b), expected_value)


#################################################################################


@parametrize_with_cases("agreement_map, expected_df", glob="crosstab_2d_DataArrays")
def test_crosstab_2d_DataArrays(agreement_map, expected_df):
    """Test crosstabbing agreement DataArrays"""
    agreement_map.gval  # Initialize gval for attributes
    crosstab_df = _crosstab_2d_DataArrays(agreement_map)
    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases("agreement_map, expected_df", glob="crosstab_3d_DataArrays")
def test_crosstab_3d_DataArrays(agreement_map, expected_df):
    """Test crosstabbing agreement DataArrays"""
    agreement_map.gval  # Initialize gval for attributes
    crosstab_df = _crosstab_3d_DataArrays(agreement_map)
    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases(
    "agreement_map, expected_df", glob="crosstab_DataArrays_success"
)
def test_crosstab_DataArrays_success(agreement_map, expected_df):
    """Test crosstabbing agreement DataArrays"""
    agreement_map.gval  # Initialize gval for attributes
    crosstab_df = _crosstab_DataArrays(agreement_map)
    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases("agreement_map", glob="crosstab_DataArrays_fail")
def test_crosstab_DataArrays_fail(agreement_map):
    """Test crosstabbing agreement DataArrays"""
    agreement_map.gval  # Initialize gval for attributes
    with raises(ValueError):
        _crosstab_DataArrays(agreement_map)


@parametrize_with_cases("agreement_map, expected_df", glob="crosstab_Datasets")
def test_crosstab_Datasets(agreement_map, expected_df):
    """Test crosstabbing agreement datasets"""
    agreement_map.gval  # Initialize gval for attributes
    crosstab_df = _crosstab_Datasets(agreement_map)
    # takes band_# pattern to just #
    crosstab_df["band"] = crosstab_df["band"].apply(lambda x: x.split("_")[-1])
    crosstab_df = Crosstab_df(crosstab_df)
    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases(
    "candidate_map, benchmark_map, agreement_map, comparison_function, pairing_dict, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata",
    glob="compute_agreement_map_success",
)
def test_compute_agreement_map_success(
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
    """Tests computing of agreement xarray from two xarrays"""

    agreement_map_computed = _compute_agreement_map(
        candidate_map,
        benchmark_map,
        comparison_function,
        pairing_dict,
        allow_candidate_values=allow_candidate_values,
        allow_benchmark_values=allow_benchmark_values,
        nodata=nodata,
        encode_nodata=encode_nodata,
    )

    # Use xr.testing.assert_identical() if names and attributes need to be compared too
    xr.testing.assert_equal(agreement_map_computed, agreement_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, comparison_function, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata",
    glob="compute_agreement_map_fail",
)
def test_compute_agreement_map_fail(
    candidate_map,
    benchmark_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
    nodata,
    encode_nodata,
):
    """Tests fail computing of agreement xarray from two xarrays"""

    with raises(ValueError):
        _ = _compute_agreement_map(
            candidate_map,
            benchmark_map,
            comparison_function,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
            nodata=nodata,
            encode_nodata=encode_nodata,
        )


@parametrize_with_cases(
    "candidate_map, benchmark_map, agreement_map, comparison_function, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata",
    glob="comparison_processing_agreement_maps_success",
)
def test_comparison_processing_agreement_map_success(
    candidate_map,
    benchmark_map,
    agreement_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
    nodata,
    encode_nodata,
):
    """Tests comparison processing computing of agreement xarray from two xarrays"""

    agreement_map_computed = compare_proc.process_agreement_map(
        comparison_function=comparison_function,
        candidate_map=candidate_map,
        benchmark_map=benchmark_map,
        allow_candidate_values=allow_candidate_values,
        allow_benchmark_values=allow_benchmark_values,
        nodata=nodata,
        encode_nodata=encode_nodata,
    )

    # Use xr.testing.assert_identical() if names and attributes need to be compared too
    xr.testing.assert_equal(agreement_map_computed, agreement_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, comparison_function, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata, exception",
    glob="comparison_processing_agreement_maps_fail",
)
def test_comparison_processing_agreement_maps_fail(
    candidate_map,
    benchmark_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
    nodata,
    encode_nodata,
    exception,
):
    with raises(exception):
        _ = compare_proc.process_agreement_map(
            comparison_function=comparison_function,
            candidate_map=candidate_map,
            benchmark_map=benchmark_map,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
            nodata=nodata,
            encode_nodata=encode_nodata,
        )


def test_comparison_get_all_param():
    """tests get all params function"""

    try:
        compare_proc.get_all_parameters()
    except KeyError:
        assert False, "Signature dict not present or keys changed"


def test_get_available_functions():
    """tests get all available functions"""

    assert compare_proc.available_functions() == [
        "pairing_dict",
        "cantor",
        "szudzik",
        "difference",
    ]


@parametrize_with_cases("args, func", glob="comparison_register_function")
def test_comparison_register_function(args, func):
    """tests register func function"""

    compare_proc.register_function(**args)(func)


@parametrize_with_cases(
    "args, func, exception", glob="comparison_register_function_fail"
)
def test_comparison_register_function_fail(args, func, exception):
    """tests register func fail function"""

    with raises(exception):
        compare_proc.register_function(**args)(func)


@parametrize_with_cases("names, args, cls", glob="comparison_register_class")
def test_register_class(names, args, cls):
    """tests register class function"""

    compare_proc.register_function_class(**args)(cls)

    if [name in compare_proc.registered_functions for name in names] != [True] * len(
        names
    ):
        assert False, "Unable to register all class functions"


@parametrize_with_cases("args, cls, exception", glob="comparison_register_class_fail")
def test_comparison_register_class_fail(args, cls, exception):
    """tests register class fail function"""

    with raises(exception):
        compare_proc.register_function_class(**args)(cls)


@parametrize_with_cases("name, params", glob="comparison_get_param")
def test_comparison_get_param(name, params):
    """tests get param function"""

    _params = compare_proc.get_parameters(name)
    assert _params == params


@parametrize_with_cases("name", glob="comparison_get_param_fail")
def test_comparison_get_param_fail(name):
    """tests get param fail function"""

    with raises(KeyError):
        _ = compare_proc.get_parameters(name)
