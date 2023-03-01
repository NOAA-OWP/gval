"""
Test functionality for gval/compare.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gval.homogenize.spatial_alignment import Spatial_alignment
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
)

from config import TEST_DATA
from tests.utils import _assert_pairing_dict_equal

test_data_dir = TEST_DATA


@pytest.fixture(
    scope="function",
    params=[(-1, 0, 2), (1.09, -13, 106), (6.090, -10, 27), (-10.39023, 13, 196)],
)
def pairs_for_natural_numbers(request):
    """makes candidate"""
    yield request.param


def test_is_not_natural_number(pairs_for_natural_numbers):
    c, b, a = pairs_for_natural_numbers
    with pytest.raises(ValueError):
        _is_not_natural_number(c)
        _is_not_natural_number(b)


@pytest.fixture(
    scope="function", params=[(1, 0, 1), (1, 13, 118), (6, 0, 21), (6, 13, 203)]
)
def cantor_pair_input(request):
    """makes candidate"""
    yield request.param


def test_cantor_pair(cantor_pair_input):
    """tests cantor pairing function"""
    c, b, a = cantor_pair_input
    np.testing.assert_equal(
        cantor_pair(c, b), a
    ), "Cantor function output does not match expected value"


@pytest.fixture(
    scope="function",
    params=[
        (-1, 0, 1),
        (1, -13, 403),
        (-6, 0, 66),
        (6, -130, 37115),
        (np.nan, -130, np.nan),
    ],
)
def cantor_pair_signed_input(request):
    """makes candidate"""
    yield request.param


@pytest.mark.filterwarnings("ignore:invalid value encountered in")
def test_cantor_pair_signed(cantor_pair_signed_input):
    """
    tests cantor pairing function
    FIXME: Figure out why RuntimeWarning is occurring for test cases with np.nan.
    """
    c, b, a = cantor_pair_signed_input
    np.testing.assert_equal(
        cantor_pair_signed(c, b), a
    ), "Signed cantor function output does not match expected value"


@pytest.fixture(
    scope="function", params=[(1, 0, 2), (1, 13, 170), (6, 0, 42), (6, 13, 175)]
)
def szudzik_pair_input(request):
    """makes candidate"""
    yield request.param


def test_szudzik_pair(szudzik_pair_input):
    """tests szudzikpairing function"""
    c, b, a = szudzik_pair_input
    np.testing.assert_equal(
        szudzik_pair(c, b), a
    ), "szudzik function output does not match expected value"


@pytest.fixture(
    scope="function",
    params=[
        (-1, 0, 2),
        (1, -13, 627),
        (-6, 0, 132),
        (6, -130, 67093),
        (np.nan, -130, np.nan),
    ],
)
def szudzik_pair_signed_input(request):
    """makes candidate"""
    yield request.param


@pytest.mark.filterwarnings("ignore:invalid value encountered in")
def test_szudzik_pair_signed(szudzik_pair_signed_input):
    """
    Tests szudzik pairing function
    FIXME: Figure out why RuntimeWarning is occurring for test cases with np.nan.
    """
    c, b, a = szudzik_pair_signed_input
    np.testing.assert_equal(
        szudzik_pair_signed(c, b), a
    ), "Signed szudzik function output does not match expected value"


@pytest.mark.parametrize(
    "unique_candidate_values, unique_benchmark_values, expected_dict",
    [
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
    ],
)
def test_make_pairing_dict(
    unique_candidate_values, unique_benchmark_values, expected_dict
):
    """Tests creation of a pairing dictionary"""

    computed_dict = _make_pairing_dict(unique_candidate_values, unique_benchmark_values)

    # breakpoint()
    _assert_pairing_dict_equal(computed_dict, expected_dict)


@pytest.mark.parametrize(
    "c, b, pairing_dict, expected_value",
    [
        (1, 2, {(1, 2): 3}, 3),
        (9, 10, {(9, 10.0): 1}, 1),
        (-1, 10, {(-1, 10): np.nan}, np.nan),
    ],
)
def test_pairing_dict_fn(c, b, pairing_dict, expected_value):
    """Tests pairing dictionary function"""

    # pairing_dict = _convert_dict_to_numba(pairing_dict)
    computed_value = pairing_dict_fn(c, b, pairing_dict)

    np.testing.assert_equal(computed_value, expected_value)


#################################################################################


@pytest.mark.parametrize(
    "expected_df",
    [
        (
            pd.DataFrame(
                {
                    "zone": [-9999, 1, 2],
                    0: [60420648, 41687643, 2232777],
                    2: [143467, 2163307, 10641990],
                },
            )
        )
    ],
)
def test_crosstab_xarray(candidate_map, benchmark_map, expected_df):
    """Test crosstabbing candidate and benchmark xarrays"""

    # TODO: make inputs that are already spatially aligned as to avoid aligning
    candidate_map, benchmark_map = Spatial_alignment(
        candidate_map, benchmark_map, "candidate"
    )

    crosstab_df = crosstab_xarray(candidate_map, benchmark_map)

    expected_df = _reorganize_crosstab_output(expected_df)

    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)


@pytest.fixture(
    scope="session",
    params=[
        (szudzik_pair_signed, "szudzik", None, None),
        (cantor_pair_signed, "cantor", None, None),
        ("pairing_dict", "pairing_dict", [-9999, 1, 2], [0, 2]),
    ],
)
def comparison(request):
    """return agreement map key"""
    yield request.param


def test_compute_agreement_xarray(
    candidate_map, benchmark_map, agreement_map, comparison
):
    """Tests computing of agreement xarray from two xarrays"""

    # TODO: make inputs that are already spatially aligned as to avoid aligning
    candidate_map, benchmark_map = Spatial_alignment(
        candidate_map, benchmark_map, "candidate"
    )

    # TODO: is there a more elegant way of parameterizing this test?
    comparison_function, _, allow_candidate_values, allow_benchmark_values = comparison

    agreement_map_computed = compute_agreement_xarray(
        candidate_map,
        benchmark_map,
        comparison_function,
        allow_candidate_values=allow_candidate_values,
        allow_benchmark_values=allow_benchmark_values,
    )

    # Use xr.testing.assert_identical() if names and attributes need to be compared too
    xr.testing.assert_equal(agreement_map_computed, agreement_map)
