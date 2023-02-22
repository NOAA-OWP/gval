"""
Test functionality for gval/compare.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import rioxarray as rxr

from gval.utils.loading_datasets import load_raster_as_xarray
from gval.prep_comparison.spatial_alignment import Spatial_alignment
from gval.compare import (
    _are_not_natural_numbers,
    cantor_pair,
    cantor_pair_signed,
    szudzik_pair,
    szudzik_pair_signed,
    compute_agreement_xarray,
    crosstab_xarray,
)

from config import TEST_DATA, AWS_KEYS
from tests.utils import _set_aws_environment_variables

test_data_dir = TEST_DATA

# set AWS environment variables
_set_aws_environment_variables(AWS_KEYS)


@pytest.fixture(
    scope="function",
    params=[(-1, 0, 2), (1.09, -13, 106), (6.090, -10, 27), (-10.39023, 13, 196)],
)
def pairs_for_natural_numbers(request):
    """makes candidate"""
    yield request.param


def test_are_not_natural_numbers(pairs_for_natural_numbers):
    c, b, a = pairs_for_natural_numbers
    with pytest.raises(ValueError):
        _are_not_natural_numbers(c, b)


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


def test_cantor_pair_signed(cantor_pair_signed_input):
    """tests cantor pairing function"""
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


def test_szudzik_pair_signed(szudzik_pair_signed_input):
    """tests szudzik pairing function"""
    c, b, a = szudzik_pair_signed_input
    np.testing.assert_equal(
        szudzik_pair_signed(c, b), a
    ), "Signed szudzik function output does not match expected value"


#################################################################################
# TODO this is duplicated across unit tests.


@pytest.fixture(scope="package", params=range(1))
def candidate_map_fp(request):
    """returns candidate maps"""
    filepath = os.path.join(test_data_dir, f"candidate_map_{request.param}.tif")
    yield filepath


@pytest.fixture(scope="package", params=range(1))
def benchmark_map_fp(request):
    """returns benchmark maps"""
    filepath = os.path.join(test_data_dir, f"benchmark_map_{request.param}.tif")
    yield filepath


@pytest.fixture(scope="package")
def candidate_map(candidate_map_fp):
    """returns candidate maps"""
    yield load_raster_as_xarray(candidate_map_fp)


@pytest.fixture(scope="package")
def benchmark_map(benchmark_map_fp):
    """returns benchmark maps"""
    yield load_raster_as_xarray(benchmark_map_fp)


#################################################################################


@pytest.mark.parametrize(
    "comparison_function, expected_df",
    [
        (
            szudzik_pair_signed,
            pd.DataFrame(
                {
                    "zone": [-9999, 1, 2],
                    0: [60420648, 41687643, 2232777],
                    2: [143467, 2163307, 10641990],
                }
            ),
        )
    ],
)
def test_crosstab_xarray(
    candidate_map, benchmark_map, comparison_function, expected_df
):
    """Test crosstabbing candidate and benchmark xarrays"""

    # todo make inputs that are already spatially aligned as to avoid aligning
    candidate_map, benchmark_map = Spatial_alignment(
        candidate_map, benchmark_map, "candidate"
    )

    crosstab_df = crosstab_xarray(candidate_map, benchmark_map, comparison_function)

    pd.testing.assert_frame_equal(crosstab_df, expected_df, check_dtype=False)
