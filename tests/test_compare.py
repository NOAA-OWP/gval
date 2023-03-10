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

from gval.compare import (
    _is_not_natural_number,
    cantor_pair,
    cantor_pair_signed,
    szudzik_pair,
    szudzik_pair_signed,
    _make_pairing_dict,
    pairing_dict_fn,
    compute_agreement_xarray,
    _reorganize_crosstab_output,
    crosstab_xarray,
    _convert_dict_to_numba,
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


@parametrize_with_cases("py_dict, numba_dict", glob="convert_dict_to_numba")
def test_convert_dict_to_numba(py_dict, numba_dict):
    """Tests converting a python dictionary to a numba dictionary"""

    nb_dict = _convert_dict_to_numba(py_dict=py_dict)
    assert nb_dict == numba_dict


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
    "candidate_map, benchmark_map, expected_df", glob="crosstab_xarray"
)
def test_crosstab_xarray(candidate_map, benchmark_map, expected_df):
    """Test crosstabbing candidate and benchmark xarrays"""

    crosstab_df = crosstab_xarray(candidate_map, benchmark_map)

    expected_df = _reorganize_crosstab_output(expected_df)

    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@parametrize_with_cases(
    "candidate_map, benchmark_map, agreement_map, comparison_function, allow_candidate_values, allow_benchmark_values",
    glob="compute_agreement_xarray",
)
def test_compute_agreement_xarray(
    candidate_map,
    benchmark_map,
    agreement_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
):
    """Tests computing of agreement xarray from two xarrays"""

    agreement_map_computed = compute_agreement_xarray(
        candidate_map,
        benchmark_map,
        comparison_function,
        allow_candidate_values=allow_candidate_values,
        allow_benchmark_values=allow_benchmark_values,
    )

    # Use xr.testing.assert_identical() if names and attributes need to be compared too
    xr.testing.assert_equal(agreement_map_computed, agreement_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, comparison_function, allow_candidate_values, allow_benchmark_values",
    glob="compute_agreement_xarray_fail",
)
def test_compute_agreement_xarray_fail(
    candidate_map,
    benchmark_map,
    comparison_function,
    allow_candidate_values,
    allow_benchmark_values,
):
    """Tests fail computing of agreement xarray from two xarrays"""

    with raises(ValueError):
        _ = compute_agreement_xarray(
            candidate_map,
            benchmark_map,
            comparison_function,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
        )
