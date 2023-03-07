"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest_cases import parametrize_with_cases
import xarray as xr

from gval.homogenize.spatial_alignment import (
    matching_crs,
    matching_spatial_indices,
    transform_bounds,
    rasters_intersect,
    align_rasters,
    Spatial_alignment,
)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_match", glob="matching_crs"
)
def test_matching_crs(candidate_map, benchmark_map, expected_match):
    """Tests if two maps have matching CRSs"""
    matching = matching_crs(candidate_map, benchmark_map)

    assert matching == expected_match, "CRSs of maps are not matching"


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_spatial_indices_matches",
    glob="matching_spatial_indices",
)
def test_matching_spatial_indices(
    candidate_map, benchmark_map, expected_spatial_indices_matches
):
    """Tests for matching indices in two xarrays"""
    matching = matching_spatial_indices(candidate_map, benchmark_map)
    assert matching == expected_spatial_indices_matches, "Indices don't match expected"


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_crs", glob="transform_bounds"
)
def test_transform_bounds(candidate_map, benchmark_map, target_crs):
    """Tests the transformation of bounds given a target crs"""
    # TODO: This test is not correct. Need to document the correct bounds and compare to those.
    tb = transform_bounds(candidate_map, benchmark_map, target_crs)
    assert isinstance(tb, tuple), f"{tb} is not tuple type"


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_intersect", glob="rasters_intersect"
)
def test_rasters_intersect(candidate_map, benchmark_map, expected_intersect):
    """Tests the intersection of rasters"""
    intersect = rasters_intersect(
        candidate_map.rio.bounds(), benchmark_map.rio.bounds()
    )
    assert intersect == expected_intersect


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs", glob="align_rasters"
)
def test_align_rasters(candidate_map, benchmark_map, target_map, **kwargs):
    """Tests the alignment of rasters"""

    cam, bem = align_rasters(candidate_map, benchmark_map, target_map, **kwargs)

    try:
        # xr.align raises a value error if coordinates don't align
        xr.align(cam, bem, join="exact")
    except ValueError:
        assert False, "Candidate and benchmark failed to align"


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs", glob="align_rasters"
)
def test_spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs):
    """Tests spatial_alignment function"""
    cam, bem = Spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs)

    try:
        # xr.align raises a value error if coordinates don't align
        xr.align(cam, bem, join="exact")
    except ValueError:
        assert False, "Candidate and benchmark failed to align"
