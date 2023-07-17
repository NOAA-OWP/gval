"""
Test cases for test_homogenize.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from itertools import product

import numpy as np
import xarray as xr
from pytest_cases import parametrize

from tests.conftest import _load_xarray, _load_gpkg

"""
Candidate Attributes
--------------------
AREA_OR_POINT=Area
test1=1
test2=2
test3=a3

Benchmark Attributes
--------------------
AREA_OR_POINT=Area
test1=1
test2=2
test3=a3
test4=4.1 
"""

candidate_map_fns = np.array(["candidate_map_attrs_1.tif"]*14)
benchmark_map_fns = np.array(["benchmark_map_attrs_1.tif"]*14)

agreement_maps = np.array([xr.DataArray(np.ones((3, 3)), dims=["y", "x"])] + [None]*13)),
candidate_suffixes = np.array(["_candidate", "_c"] + ["_candidate"]*12)
benchmark_suffixes = np.array(["_benchmark", "_b"] + ["_benchmark"]*12),
candidate_includes = np.array([None, None, ("test1","test2")] + [None]*11),
candidate_excludes = np.array([None, None, None, ("test2")] + [None]*10),
benchmark_includes = np.array([None, None, None, None, ("test1","test4")] + [None]*9),
benchmark_excludes = np.array([None, None, None, None, None, ("test1","test2")] + [None]*8),


expected_dfs = [
    pd.DataFrame(
        {
            "AREA_OR_POINT_candidate":"Area",
            "test1_candidate":1,
            "test2_candidate":2,
            "test3_candidate":"a3",
            "AREA_OR_POINT_benchmark":"Area",
            "test1_benchmark":1,
            "test2_benchmark":2,
            "test3_benchmark":"a3",
            "test4_benchmark":4.1
        },
    ),
    pd.DataFrame(
        {
            "AREA_OR_POINT_c":"Area",
            "test1_c":1,
            "test2_c":2,
            "test3_c":"a3",
            "AREA_OR_POINT_b":"Area",
            "test1_b":1,
            "test2_b":2,
            "test3_b":"a3",
            "test4_b":4.1
        },
    ),
    pd.DataFrame(
        {
            "test1_candidate":1,
            "test2_candidate":2,
            "AREA_OR_POINT_benchmark":"Area",
            "test1_benchmark":1,
            "test2_benchmark":2,
            "test3_benchmark":"a3",
            "test4_benchmark":4.1
        },
    ),
    pd.DataFrame(
        {
            "AREA_OR_POINT_candidate":"Area",
            "test1_candidate":1,
            "test3_candidate":"a3",
            "AREA_OR_POINT_benchmark":"Area",
            "test1_benchmark":1,
            "test2_benchmark":2,
            "test3_benchmark":"a3",
            "test4_benchmark":4.1
        },
    ),
    pd.DataFrame(
        {
            "AREA_OR_POINT_candidate":"Area",
            "test1_candidate":1,
            "test2_candidate":2,
            "test3_candidate":"a3",
            "test1_benchmark":1,
            "test4_benchmark":4.1
        },
    ),
    pd.DataFrame(
        {
            "AREA_OR_POINT_candidate":"Area",
            "test1_candidate":1,
            "test2_candidate":2,
            "test3_candidate":"a3",
            "AREA_OR_POINT_benchmark":"Area",
            "test3_benchmark":"a3",
            "test4_benchmark":4.1
        },
    ),
]

expected_attrs = np.array(
    [{
        "AREA_OR_POINT_candidate":"Area",
        "test1_candidate":1,
        "test2_candidate":2,
        "test3_candidate":"a3",
        "AREA_OR_POINT_benchmark":"Area",
        "test1_benchmark":1,
        "test2_benchmark":2,
        "test3_benchmark":"a3",
        "test4_benchmark":4.1
    }] + [None]*13
)

@parametrize(
    "candidate_map_fn, benchmark_map_fn, agreement_map, candidate_suffix, benchmark_suffix, candidate_include, candidate_exclude, benchmark_include, benchmark_exclude, expected_df, expected_attr",
    list(zip(candidate_map_fns, benchmark_map_fns, agreement_maps, candidate_suffixes, benchmark_suffixes, candidate_includes, candidate_excludes, benchmark_includes, benchmark_excludes, expected_dfs))
)
def case_attribute_tracking(
    candidate_map_fn,
    benchmark_map_fn,
    agreement_map,
    candidate_suffix,
    benchmark_suffix,
    candidate_include,
    candidate_exclude,
    benchmark_include,
    benchmark_exclude,
    expected_df,
    expected_attr
):
    return (
        _load_xarray(candidate_map_fn),
        _load_xarray(benchmark_map_fn),
        agreement_map,
        candidate_suffix,
        benchmark_suffix,
        candidate_include,
        candidate_exclude,
        benchmark_include,
        benchmark_exclude,
        expected_df,
        expected_attr
    )