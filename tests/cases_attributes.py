"""
Test cases for test_homogenize.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from itertools import product

import numpy as np
from pytest_cases import parametrize

from tests.conftest import _load_xarray, _load_gpkg

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