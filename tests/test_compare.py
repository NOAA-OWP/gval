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
    pairing_dict_fn,
)
from gval.comparison.agreement import _compute_agreement_map
from gval.comparison.tabulation import (
    _convert_crosstab_to_contigency_table,
    _crosstab_2d_DataArrays,
    _crosstab_3d_DataArrays,
    _crosstab_DataArrays,
    _crosstab_Datasets,
)
from tests.conftest import _assert_pairing_dict_equal


@parametrize_with_cases("number", glob="is_not_natural_number_successes")
def test_is_not_natural_number_success(number):
    _is_not_natural_number(number)


@parametrize_with_cases("number", glob="is_not_natural_number_failures")
def test_is_not_natural_number_failures(number):
    with pytest.raises(ValueError):
        _is_not_natural_number(number)


@parametrize_with_cases("c, b, a", glob="cantor_pair")
def test_cantor_pair(c, b, a):
    """tests cantor pairing function"""
    np.testing.assert_equal(
        cantor_pair(c, b), a
    ), "Cantor function output does not match expected value"


@pytest.mark.filterwarnings("ignore:invalid value encountered in")
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


@pytest.mark.filterwarnings("ignore:invalid value encountered in")
@parametrize_with_cases("c, b, a", glob="szudzik_pair_signed")
def test_szudzik_pair_signed(c, b, a):
    """
    Tests szudzik pairing function
    FIXME: Figure out why RuntimeWarning is occurring for test cases with np.nan.
    """
    np.testing.assert_equal(
        szudzik_pair_signed(c, b), a
    ), "Signed szudzik function output does not match expected value"


# @parametrize_with_cases("py_dict, numba_dict", glob="convert_dict_to_numba")
# def test_convert_dict_to_numba(py_dict, numba_dict):
#     """Tests converting a python dictionary to a numba dictionary"""
#
#     nb_dict = _convert_dict_to_numba(py_dict=py_dict)
#     assert nb_dict == numba_dict


@parametrize_with_cases(
    "unique_candidate_values, unique_benchmark_values, expected_dict",
    glob="make_pairing_dict",
)
def test_make_pairing_dict(
    unique_candidate_values, unique_benchmark_values, expected_dict
):
    """Tests creation of a pairing dictionary"""

    computed_dict = _make_pairing_dict(unique_candidate_values, unique_benchmark_values)

    _assert_pairing_dict_equal(computed_dict, expected_dict)


@parametrize_with_cases("c, b, pairing_dict, expected_value", glob="pairing_dict_fn")
def test_pairing_dict_fn(c, b, pairing_dict, expected_value):
    """Tests pairing dictionary function"""

    computed_value = pairing_dict_fn(c, b, pairing_dict)

    np.testing.assert_equal(computed_value, expected_value)


#################################################################################


@parametrize_with_cases(
    "crosstab_df, expected_df", glob="convert_crosstab_to_contigency_table"
)
def test_convert_crosstab_to_contigency_table(crosstab_df, expected_df):
    computed_df = _convert_crosstab_to_contigency_table(crosstab_df)
    pd.testing.assert_frame_equal(computed_df, expected_df, check_dtype=False)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_df", glob="crosstab_2d_DataArrays"
)
def test_crosstab_2d_DataArrays(candidate_map, benchmark_map, expected_df):
    """Test crosstabbing candidate and benchmark DataArrays"""
    crosstab_df = _crosstab_2d_DataArrays(candidate_map, benchmark_map)
    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_df", glob="crosstab_3d_DataArrays"
)
def test_crosstab_3d_DataArrays(candidate_map, benchmark_map, expected_df):
    """Test crosstabbing candidate and benchmark DataArrays"""
    crosstab_df = _crosstab_3d_DataArrays(candidate_map, benchmark_map)
    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_df", glob="crosstab_DataArrays_success"
)
def test_crosstab_DataArrays_success(candidate_map, benchmark_map, expected_df):
    """Test crosstabbing candidate and benchmark DataArrays"""
    crosstab_df = _crosstab_DataArrays(candidate_map, benchmark_map)
    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases("candidate_map, benchmark_map", glob="crosstab_DataArrays_fail")
def test_crosstab_DataArrays_fail(candidate_map, benchmark_map):
    """Test crosstabbing candidate and benchmark DataArrays"""
    with raises(ValueError):
        _crosstab_DataArrays(candidate_map, benchmark_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_df", glob="crosstab_Datasets"
)
def test_crosstab_Datasets(candidate_map, benchmark_map, expected_df):
    """Test crosstabbing candidate and benchmark datasets"""
    crosstab_df = _crosstab_Datasets(candidate_map, benchmark_map)
    crosstab_df["band"] = crosstab_df["band"].apply(lambda x: int(x.split("_")[-1]))
    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases(
    "candidate_map, benchmark_map, agreement_map, comparison_function, allow_candidate_values, allow_benchmark_values, nodata, encode_nodata",
    glob="compute_agreement_map_success",
)
def test_compute_agreement_map_success(
    candidate_map,
    benchmark_map,
    agreement_map,
    comparison_function,
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
