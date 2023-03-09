"""
Test cases for test_spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from itertools import product

import numpy as np
from pytest_cases import parametrize
import pyproj

from tests.conftest import _load_xarray


candidate_map_fns = np.array(["candidate_map_0.tif", "candidate_map_1.tif"])
benchmark_map_fns = np.array(["benchmark_map_0.tif", "benchmark_map_1.tif"])
target_map_fns = np.array([None, "target_map_0.tif", "target_map_1.tif"])

# TODO: Needs cases where they don't match and an argument that expects that.
expected_crs_matches = np.array([True, False])


@parametrize(
    "candidate_map_fn, benchmark_map_fn, expected_crs_matches",
    list(zip(candidate_map_fns, benchmark_map_fns, expected_crs_matches)),
)
def case_matching_crs(candidate_map_fn, benchmark_map_fn, expected_crs_matches):
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        expected_crs_matches,
    )


expected_spatial_indices_matches = [False, False]


@parametrize(
    "candidate_map_fn, benchmark_map_fn, expected_spatial_indices_match",
    list(zip(candidate_map_fns, benchmark_map_fns, expected_spatial_indices_matches)),
)
def case_matching_spatial_indices(
    candidate_map_fn, benchmark_map_fn, expected_spatial_indices_match
):
    # TODO: How does this do with bands?
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        expected_spatial_indices_match,
    )


target_crss = [
    "benchmark",
    "candidate",
    "EPSG:4329",
    pyproj.CRS(proj="utm", zone=10, ellps="WGS84"),
]


@parametrize(
    "candidate_map_fn, benchmark_map_fn",
    list(zip(candidate_map_fns, benchmark_map_fns)),
)
@parametrize("target_crs", target_crss)
def case_transform_bounds(candidate_map_fn, benchmark_map_fn, target_crs):
    return (_load_xarray(candidate_map_fn), _load_xarray(benchmark_map_fn), target_crs)


expected_intersections = [True, False, False, False]


@parametrize(
    "candidate_map_fn, benchmark_map_fn, expected_intersect",
    # take all combinations of candidates and benchmarks then align expected to those combinations
    [
        cb + (ei,)
        for cb, ei in zip(
            product(candidate_map_fns, benchmark_map_fns), expected_intersections
        )
    ],
)
def case_rasters_intersect(candidate_map_fn, benchmark_map_fn, expected_intersect):
    # TODO: Need to test rasters that don't spatially intersect
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        expected_intersect,
    )


kwargss = [None, {"dst_crs": "EPSG:4329"}]
target_maps = ["candidate", "benchmark"]


@parametrize(
    "candidate_map_fn, benchmark_map_fn",
    list(zip(candidate_map_fns, benchmark_map_fns)),
)
@parametrize("target_map", target_maps)
@parametrize("kwargs", kwargss)
def case_align_rasters(candidate_map_fn, benchmark_map_fn, target_map, kwargs):
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        target_map,
        kwargs,
    )
