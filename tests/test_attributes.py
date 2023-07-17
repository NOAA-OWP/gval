"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest import raises
from pytest_cases import parametrize_with_cases

from gval.attributes.attributes import _attribute_tracking_xarray

@parametrize_with_cases(
    "candidate_map, benchmark_map, agreement_map, candidate_suffix, benchmark_suffix, candidate_include, candidate_exclude, benchmark_include, benchmark_exclude, expected_df",
    glob="attribute_tracking"
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
    expected_df
):
    """Tests attribute tracking functionality"""

    # Test attribute tracking
    results = _attribute_tracking_xarray(
        candidate_map = candidate_map,
        benchmark_map = benchmark_map,
        agreement_map = agreement_map,
        candidate_suffix = candidate_suffix,
        benchmark_suffix = benchmark_suffix,
        candidate_include = candidate_include,
        candidate_exclude = candidate_exclude,
        benchmark_include = benchmark_include
    )

    if agreement_map is None:
        pd.testing.assert_frame_equal(results, expected_df)
    else:
        pd.testing.assert_frame_equal(results[0], expected_df)
        xr.testing.assert_identical(results[1], expected_attr)