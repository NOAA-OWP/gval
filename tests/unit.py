"""
Test functionality
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import pytest
import os
import sys
import xarray

from gval.utils.loading_datasets import load_raster_as_xarray
from gval.prep_comparison.spatial_alignment import (
    matching_crs,
    checks_for_single_band,
    matching_spatial_indices,
    transform_bounds,
    rasters_intersect,
    align_rasters,
    Spatial_alignment,
)
from gval.compare import crosstab_rasters
from config import TEST_DATA

test_data_dir = TEST_DATA

# temporary
sys.path.append(os.path.abspath(".."))


@pytest.fixture(scope="module", params=range(1))
def candidate_map_fp(request):
    """returns candidate maps"""
    filepath = os.path.join(test_data_dir, f"candidate_map_{request.param}.tif")
    yield filepath


@pytest.fixture(scope="module", params=range(1))
def benchmark_map_fp(request):
    """returns benchmark maps"""
    filepath = os.path.join(test_data_dir, f"benchmark_map_{request.param}.tif")
    yield filepath


@pytest.fixture(scope="module")
def candidate_map(candidate_map_fp):
    """returns candidate maps"""
    yield load_raster_as_xarray(candidate_map_fp)


@pytest.fixture(scope="module")
def benchmark_map(benchmark_map_fp):
    """returns benchmark maps"""
    yield load_raster_as_xarray(benchmark_map_fp)


def test_load_candidate_as_xarray(candidate_map_fp):
    """tests loading candidate raster as xarray DataArray"""
    candidate_map = load_raster_as_xarray(candidate_map_fp)
    assert isinstance(
        candidate_map, xarray.DataArray
    ), "candidate_map is not an xarray.DataArray"


def test_load_benchmark_as_xarray(benchmark_map_fp):
    """tests loading benchmark raster as xarray DataArray"""
    benchmark_map = load_raster_as_xarray(benchmark_map_fp)
    assert isinstance(
        benchmark_map, xarray.DataArray
    ), "benchmark_map is not an xarray.DataArray"


@pytest.fixture(scope="module", params=[True])
def expect_matching_crs(request):
    """Returns expect value for matching CRS test"""
    yield request.param


def test_matching_crs(candidate_map, benchmark_map, expect_matching_crs):
    """Tests for matching CRSs"""
    matching = matching_crs(candidate_map, benchmark_map)
    assert (
        matching == expect_matching_crs
    ), f"matching_crs result ({matching}) does not agree with expected ({expect_matching_crs})"


@pytest.fixture(scope="function", params=[False])
def expect_matching_spatial_indices(request):
    """Returns expect value for matching indices test"""
    yield request.param


def test_matching_spatial_indices(
    candidate_map, benchmark_map, expect_matching_spatial_indices
):
    """Tests for matching indices in two xarray DataArrays"""
    matching = matching_spatial_indices(candidate_map, benchmark_map)
    assert (
        matching == expect_matching_spatial_indices
    ), f"Expected {expect_matching_spatial_indices} while matching indices {matching}"


@pytest.fixture(scope="module", params=["benchmark", "candidate"])
def target_map(request):
    """Target map fixture"""
    yield request.param


@pytest.fixture(scope="module", params=[None, "EPSG:4329"])
def dst_crs(request):
    """dst_crs fixture"""
    yield request.param


def test_transform_bounds(candidate_map, benchmark_map, target_map, dst_crs):
    """Tests the transformation of bounds given a target map or dst_crs"""
    tb = transform_bounds(candidate_map, benchmark_map, target_map, dst_crs)
    assert isinstance(tb, tuple), f"{tb} is not tuple type"


def test_rasters_intersect(candidate_map, benchmark_map):
    """Tests the intersection of rasters"""
    intersect = rasters_intersect(
        candidate_map.rio.bounds(), benchmark_map.rio.bounds()
    )
    assert intersect, "Maps don't spatially intersect"


@pytest.fixture(scope="function", params=[True])
def expect_checks_for_single_band(request):
    """checks for single band expects fixture"""
    yield request.param


def test_checks_for_single_band(
    candidate_map, benchmark_map, expect_checks_for_single_band
):
    """Tests checks for single band fixture"""
    bsb = checks_for_single_band(candidate_map, benchmark_map)
    assert (
        bsb == expect_checks_for_single_band
    ), "Both candidate and benchmark expected to be single band"


@pytest.fixture(scope="module", params=[None, {"dst_crs": "EPSG:4329"}])
def kwargs(request):
    """kwargs fixture"""
    yield request.param


def test_align_rasters(candidate_map, benchmark_map, target_map, **kwargs):
    """Tests the alignment of rasters"""
    cam, bem = align_rasters(candidate_map, benchmark_map, target_map, **kwargs)
    assert isinstance(
        cam, xarray.DataArray
    ), "Aligned candidate raster not xarray DataArray"
    assert isinstance(
        bem, xarray.DataArray
    ), "Aligned benchmark raster not xarray DataArray"


def test_spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs):
    """Tests spatial_alignment function"""
    cam, bem = Spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs)
    assert isinstance(
        cam, xarray.DataArray
    ), "Aligned candidate raster not xarray DataArray"
    assert isinstance(
        bem, xarray.DataArray
    ), "Aligned benchmark raster not xarray DataArray"


def test_crosstab_rasters(candidate_map, benchmark_map):
    """Test crosstabulation of rasters"""
    cam, bem = Spatial_alignment(candidate_map, benchmark_map, "benchmark")
    output = crosstab_rasters(candidate_map, benchmark_map)
    am, ct = output.compute()
    assert isinstance(am, xarray.DataArray), "Agreement map is not xarray DataArray"
