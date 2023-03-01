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
    compute_agreement_xarray,
    _reorganize_crosstab_output,
    crosstab_xarray,
)

from config import TEST_DATA

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
    params=[(szudzik_pair_signed, "szudzik"), (cantor_pair_signed, "cantor")],
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

    comparison_function, _ = comparison
    agreement_map_computed = compute_agreement_xarray(
        candidate_map, benchmark_map, comparison_function
    )

    # Use xr.testing.assert_identical() if names and attributes need to be compared too
    xr.testing.assert_equal(agreement_map_computed, agreement_map)
