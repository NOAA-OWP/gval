"""
Test cases for test_spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from itertools import product

import numpy as np
from pytest_cases import parametrize
from rasterio.enums import Resampling

from tests.conftest import _load_xarray

candidate_map_fns = np.array(["candidate_map_0.tif", "candidate_map_1.tif"])
benchmark_map_fns = np.array(["benchmark_map_0.tif", "benchmark_map_1.tif"])

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


expected_spatial_indices_matches = [True, False]


@parametrize(
    "candidate_map_fn, benchmark_map_fn, expected_spatial_indices_match",
    list(zip(candidate_map_fns, benchmark_map_fns, expected_spatial_indices_matches)),
)
def case_matching_spatial_indices_success(
    candidate_map_fn, benchmark_map_fn, expected_spatial_indices_match
):
    # TODO: How does this do with bands?
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        expected_spatial_indices_match,
    )


@parametrize(
    "candidate_map_fn, benchmark_map_fn, expected_spatial_indices_match",
    [(candidate_map_fns[1], benchmark_map_fns[1], expected_spatial_indices_matches[1])],
)
def case_matching_spatial_indices_fail(
    candidate_map_fn, benchmark_map_fn, expected_spatial_indices_match
):
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        expected_spatial_indices_match,
    )


expected_intersections = [True, True, False, True]
# take all combinations of candidates and benchmarks then align expected to those combinations
raster_intersections = [
    cb + (ei,)
    for cb, ei in zip(
        product(candidate_map_fns, benchmark_map_fns), expected_intersections
    )
]


@parametrize(
    "candidate_map_fn, benchmark_map_fn, expected_intersect",
    raster_intersections,
)
def case_rasters_intersect_no_exception(
    candidate_map_fn, benchmark_map_fn, expected_intersect
):
    # TODO: Need to test rasters that don't spatially intersect
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        expected_intersect,
    )


@parametrize(
    "candidate_map_fn, benchmark_map_fn, expected_intersect",
    [raster_intersections[2]],
)
def case_rasters_intersect_exception(
    candidate_map_fn, benchmark_map_fn, expected_intersect
):
    # TODO: Need to test rasters that don't spatially intersect
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        expected_intersect,
    )


@parametrize(
    "candidate_map_fn, benchmark_map_fn, resampling, target_map",
    list(
        zip(
            candidate_map_fns[[0, 1, 1, 1, 1]],
            benchmark_map_fns[[0, 1, 1, 1, 1]],
            [{}, {}, {}, {}, {"resampling": Resampling.bilinear}],
            [
                "candidate",
                "benchmark",
                _load_xarray("target_map_0.tif"),
                "candidate",
                "candidate",
            ],
        )
    ),
)
def case_align_rasters(candidate_map_fn, benchmark_map_fn, target_map, resampling):
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        target_map,
        resampling,
    )


@parametrize(
    "candidate_map_fn, benchmark_map_fn, resampling, target_map",
    list(
        zip(
            candidate_map_fns[[1, 1]],
            benchmark_map_fns[[1, 1]],
            [{}, {}],
            [None, 4.3],
        )
    ),
)
def case_align_rasters_fail(candidate_map_fn, benchmark_map_fn, target_map, resampling):
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        target_map,
        resampling,
    )


def make_no_data_value_dataset():
    dataset = _load_xarray("benchmark_map_1.tif", band_as_variable=True)
    dataset["band_1"] = dataset["band_1"].rio.write_nodata(None)
    return dataset


@parametrize(
    "candidate_map, benchmark_map, resampling, target_map",
    list(
        zip(
            [
                _load_xarray("candidate_map_0.tif").rio.write_nodata(
                    None, inplace=True
                ),
                _load_xarray("candidate_map_0.tif").rio.write_nodata(
                    -9999, inplace=True
                ),
            ],
            [_load_xarray("benchmark_map_1.tif"), make_no_data_value_dataset()],
            [{}, {}],
            ["candidate", "candidate"],
        )
    ),
)
def case_align_rasters_fail_nodata(
    candidate_map, benchmark_map, target_map, resampling
):
    return (
        candidate_map,
        benchmark_map,
        target_map,
        resampling,
    )


@parametrize(
    "candidate_map_fn, benchmark_map_fn, resampling, target_map",
    list(
        zip(
            candidate_map_fns,
            benchmark_map_fns,
            [{}, {"resampling": Resampling.nearest}],
            ["benchmark", "candidate"],
        )
    ),
)
def case_spatial_alignment(candidate_map_fn, benchmark_map_fn, resampling, target_map):
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        target_map,
        resampling,
    )


@parametrize(
    "candidate_map_fn, benchmark_map_fn, resampling, target_map",
    list(
        zip(
            candidate_map_fns[[1]],
            benchmark_map_fns[[0]],
            [{}],
            ["candidate"],
        )
    ),
)
def case_spatial_alignment_fail(
    candidate_map_fn, benchmark_map_fn, target_map, resampling
):
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        target_map,
        resampling,
    )
