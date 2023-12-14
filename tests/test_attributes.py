"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest import raises
from pytest_cases import parametrize_with_cases
import pandas as pd

from gval.attributes.attributes import _attribute_tracking_xarray
from tests.conftest import _assert_pairing_dict_equal


@parametrize_with_cases(
    "candidate_map, benchmark_map, agreement_map, candidate_suffix, benchmark_suffix, candidate_include, candidate_exclude, benchmark_include, benchmark_exclude, expected_df, expected_attr",
    glob="attribute_tracking",
)
def test_attribute_tracking(
    candidate_map,
    benchmark_map,
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
    """Tests attribute tracking functionality"""

    # Test attribute tracking
    results = _attribute_tracking_xarray(
        candidate_map=candidate_map,
        benchmark_map=benchmark_map,
        agreement_map=agreement_map,
        candidate_suffix=candidate_suffix,
        benchmark_suffix=benchmark_suffix,
        candidate_include=candidate_include,
        candidate_exclude=candidate_exclude,
        benchmark_include=benchmark_include,
        benchmark_exclude=benchmark_exclude,
    )

    if agreement_map is None:
        pd.testing.assert_frame_equal(results, expected_df)
    else:
        pd.testing.assert_frame_equal(results[0], expected_df)
        _assert_pairing_dict_equal(results[1].attrs, expected_attr)


@parametrize_with_cases(
    "candidate_map, benchmark_map, candidate_include, candidate_exclude, benchmark_include, benchmark_exclude, exception",
    glob="attribute_tracking_fail",
)
def test_attribute_tracking_fail(
    candidate_map,
    benchmark_map,
    candidate_include,
    candidate_exclude,
    benchmark_include,
    benchmark_exclude,
    exception,
):
    """Tests attribute tracking functionality"""

    # Test attribute tracking
    with raises(exception):
        _attribute_tracking_xarray(
            candidate_map=candidate_map,
            benchmark_map=benchmark_map,
            candidate_include=candidate_include,
            candidate_exclude=candidate_exclude,
            benchmark_include=benchmark_include,
            benchmark_exclude=benchmark_exclude,
        )
