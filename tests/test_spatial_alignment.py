"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest import raises
from pytest_cases import parametrize_with_cases
import xarray as xr

from gval.homogenize.spatial_alignment import (
    _matching_crs,
    _matching_spatial_indices,
    _rasters_intersect,
    _align_rasters,
    _spatial_alignment,
)
from gval.homogenize.rasterize import _rasterize_data
from gval.utils.exceptions import RasterMisalignment, RastersDontIntersect


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_match", glob="matching_crs"
)
def test_matching_crs(candidate_map, benchmark_map, expected_match):
    """Tests if two maps have matching CRSs"""
    matching = _matching_crs(candidate_map, benchmark_map)

    assert matching == expected_match, "CRSs of maps are not matching"


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_spatial_indices_matches",
    glob="matching_spatial_indices_success",
)
def test_matching_spatial_indices_success(
    candidate_map, benchmark_map, expected_spatial_indices_matches
):
    """Tests for matching indices in two xarrays"""
    matching = _matching_spatial_indices(candidate_map, benchmark_map)
    assert matching == expected_spatial_indices_matches, "Indices don't match expected"


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_spatial_indices_matches",
    glob="matching_spatial_indices_fail",
)
def test_matching_spatial_indices_fail(
    candidate_map, benchmark_map, expected_spatial_indices_matches
):
    """Tests for matching indices in two xarrays"""
    with raises(RasterMisalignment):
        _matching_spatial_indices(candidate_map, benchmark_map, raise_exception=True)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_intersect",
    glob="rasters_intersect_no_exception",
)
def test_rasters_intersect_no_exception(
    candidate_map, benchmark_map, expected_intersect
):
    """Tests the intersection of rasters"""
    intersect = _rasters_intersect(candidate_map, benchmark_map)
    assert intersect == expected_intersect


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_intersect",
    glob="rasters_intersect_exception",
)
def test_rasters_intersect_exception(candidate_map, benchmark_map, expected_intersect):
    """Tests the intersection of rasters"""
    with raises(RastersDontIntersect):
        _rasters_intersect(candidate_map, benchmark_map, raise_exception=True)


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs", glob="align_rasters"
)
def test_align_rasters(candidate_map, benchmark_map, target_map, kwargs):
    """Tests the alignment of rasters"""

    # This might raise value errors associated with
    cam, bem = _align_rasters(candidate_map, benchmark_map, target_map, **kwargs)

    # this tests for matching spatial indices
    _matching_spatial_indices(cam, bem, raise_exception=True)


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs", glob="align_rasters_fail"
)
def test_align_rasters_fail(candidate_map, benchmark_map, target_map, kwargs):
    """Tests the alignment of rasters"""

    with raises(ValueError):
        _, _ = _align_rasters(candidate_map, benchmark_map, target_map, **kwargs)


@parametrize_with_cases(
    "candidate_map, benchmark_map, resampling, target_map",
    glob="align_rasters_fail_nodata",
)
def test_align_rasters_fail_nodata(
    candidate_map, benchmark_map, resampling, target_map
):
    """Tests the alignment of rasters"""

    with raises(ValueError):
        _, _ = _align_rasters(candidate_map, benchmark_map, target_map, resampling)


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs",
    glob="spatial_alignment",
)
def test_spatial_alignment(candidate_map, benchmark_map, target_map, kwargs):
    """Tests spatial_alignment function"""

    cam, bem = _spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs)

    try:
        # xr.align raises a value error if coordinates don't align
        xr.align(cam, bem, join="exact")
    except ValueError:
        assert False, "Candidate and benchmark failed to align"


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs",
    glob="spatial_alignment_fail",
)
def test_spatial_alignment_fail(candidate_map, benchmark_map, target_map, kwargs):
    """Tests spatial_alignment function"""

    with raises(RastersDontIntersect):
        _, _ = _spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs)


@parametrize_with_cases(
    "candidate_map, benchmark_map, rasterize_attributes",
    glob="rasterize_vector_success",
)
def test_rasterize_vector_success(candidate_map, benchmark_map, rasterize_attributes):
    """Test rasterize vector success"""

    benchmark_raster = _rasterize_data(
        candidate_map=candidate_map,
        benchmark_map=benchmark_map,
        rasterize_attributes=rasterize_attributes,
    )

    if isinstance(benchmark_raster, xr.Dataset):
        assert benchmark_raster.band_1.shape == candidate_map.band_1.shape
    else:
        assert benchmark_raster.shape == candidate_map.shape


@parametrize_with_cases(
    "candidate_map, benchmark_map, rasterize_attributes",
    glob="rasterize_vector_fail",
)
def test_rasterize_vector_fail(candidate_map, benchmark_map, rasterize_attributes):
    """Tests rasterize vector fail"""

    with raises(KeyError):
        _, _ = _rasterize_data(
            candidate_map=candidate_map,
            benchmark_map=benchmark_map,
            rasterize_attributes=rasterize_attributes,
        )
