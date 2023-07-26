"""
Test cases for test_homogenize.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np
import xarray as xr
import pandas as pd
from pytest_cases import parametrize

from tests.conftest import _load_xarray

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

candidate_map_fns = ["candidate_map_attrs_1.tif"] * 14
benchmark_map_fns = ["benchmark_map_attrs_1.tif"] * 14

agreement_maps = [xr.DataArray(np.ones((3, 3)), dims=["y", "x"])] + [None] * 13
candidate_suffixes = ["_candidate", "_c"] + ["_candidate"] * 12
benchmark_suffixes = ["_benchmark", "_b"] + ["_benchmark"] * 12
candidate_includes = [None, None, ["test1", "test2"]] + [None] * 11
candidate_excludes = [None, None, None, ["test2"]] + [None] * 10
benchmark_includes = [None, None, None, None, ["test1", "test4"]] + [None] * 9
benchmark_excludes = [None, None, None, None, None, ["test1", "test2"]] + [None] * 8


expected_dfs = [
    pd.DataFrame(
        {
            "AREA_OR_POINT_candidate": "Area",
            "test1_candidate": 1,
            "test2_candidate": 2,
            "test3_candidate": "a3",
            "AREA_OR_POINT_benchmark": "Area",
            "test1_benchmark": 1,
            "test2_benchmark": 2,
            "test3_benchmark": "a3",
            "test4_benchmark": 4.1,
        },
        index=[0],
    ),
    pd.DataFrame(
        {
            "AREA_OR_POINT_c": "Area",
            "test1_c": 1,
            "test2_c": 2,
            "test3_c": "a3",
            "AREA_OR_POINT_b": "Area",
            "test1_b": 1,
            "test2_b": 2,
            "test3_b": "a3",
            "test4_b": 4.1,
        },
        index=[0],
    ),
    pd.DataFrame(
        {
            "test1_candidate": 1,
            "test2_candidate": 2,
            "AREA_OR_POINT_benchmark": "Area",
            "test1_benchmark": 1,
            "test2_benchmark": 2,
            "test3_benchmark": "a3",
            "test4_benchmark": 4.1,
        },
        index=[0],
    ),
    pd.DataFrame(
        {
            "AREA_OR_POINT_candidate": "Area",
            "test1_candidate": 1,
            "test3_candidate": "a3",
            "AREA_OR_POINT_benchmark": "Area",
            "test1_benchmark": 1,
            "test2_benchmark": 2,
            "test3_benchmark": "a3",
            "test4_benchmark": 4.1,
        },
        index=[0],
    ),
    pd.DataFrame(
        {
            "AREA_OR_POINT_candidate": "Area",
            "test1_candidate": 1,
            "test2_candidate": 2,
            "test3_candidate": "a3",
            "test1_benchmark": 1,
            "test4_benchmark": 4.1,
        },
        index=[0],
    ),
    pd.DataFrame(
        {
            "AREA_OR_POINT_candidate": "Area",
            "test1_candidate": 1,
            "test2_candidate": 2,
            "test3_candidate": "a3",
            "AREA_OR_POINT_benchmark": "Area",
            "test3_benchmark": "a3",
            "test4_benchmark": 4.1,
        },
        index=[0],
    ),
]

expected_attrs = np.array(
    [
        {
            "AREA_OR_POINT_candidate": "Area",
            "test1_candidate": 1,
            "test2_candidate": 2,
            "test3_candidate": "a3",
            "AREA_OR_POINT_benchmark": "Area",
            "test1_benchmark": 1,
            "test2_benchmark": 2,
            "test3_benchmark": "a3",
            "test4_benchmark": 4.1,
        }
    ]
    + [None] * 13
)


@parametrize(
    "candidate_map_fn, benchmark_map_fn, agreement_map, candidate_suffix, benchmark_suffix, candidate_include, candidate_exclude, benchmark_include, benchmark_exclude, expected_df, expected_attr",
    list(
        zip(
            candidate_map_fns,
            benchmark_map_fns,
            agreement_maps,
            candidate_suffixes,
            benchmark_suffixes,
            candidate_includes,
            candidate_excludes,
            benchmark_includes,
            benchmark_excludes,
            expected_dfs,
            expected_attrs,
        )
    ),
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
    expected_attr,
):
    return (
        _load_xarray(candidate_map_fn, band_as_variable=True, mask_and_scale=True),
        _load_xarray(benchmark_map_fn, band_as_variable=True, mask_and_scale=True),
        agreement_map,
        candidate_suffix,
        benchmark_suffix,
        candidate_include,
        candidate_exclude,
        benchmark_include,
        benchmark_exclude,
        expected_df,
        expected_attr,
    )


candidate_map_fns = candidate_map_fns[0:2]
benchmark_map_fns = benchmark_map_fns[0:2]

candidate_includes = [["test1", "test2"], None]
candidate_excludes = [["test2"], None]

benchmark_includes = [None, ["test1", "test4"]]
benchmark_excludes = [None, ["test1", "test2"]]

exceptions = [ValueError, ValueError]


@parametrize(
    "candidate_map_fn, benchmark_map_fn, candidate_include, candidate_exclude, benchmark_include, benchmark_exclude, exception",
    list(
        zip(
            candidate_map_fns,
            benchmark_map_fns,
            candidate_includes,
            candidate_excludes,
            benchmark_includes,
            benchmark_excludes,
            exceptions,
        )
    ),
)
def case_attribute_tracking_fail(
    candidate_map_fn,
    benchmark_map_fn,
    candidate_include,
    candidate_exclude,
    benchmark_include,
    benchmark_exclude,
    exception,
):
    return (
        _load_xarray(candidate_map_fn, band_as_variable=True),
        _load_xarray(benchmark_map_fn, band_as_variable=True),
        candidate_include,
        candidate_exclude,
        benchmark_include,
        benchmark_exclude,
        exception,
    )
