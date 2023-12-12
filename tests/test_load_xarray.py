"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest_cases import parametrize_with_cases
import xarray as xr
from pytest import raises

from gval.utils.loading_datasets import _create_xarray, _create_xarray_pairs


@parametrize_with_cases(
    "upper_left, lower_right, sizes, band_params, nodata_value, encoded_nodata_value, shapes, band_dim_name, return_dataset, expected_xr",
    glob="create_xarray_*_success",
)
def test_create_xarray(
    upper_left,
    lower_right,
    sizes,
    band_params,
    nodata_value,
    encoded_nodata_value,
    shapes,
    band_dim_name,
    return_dataset,
    expected_xr,
):
    """tests creating xarray from scratch"""

    computed_xr = _create_xarray(
        upper_left,
        lower_right,
        sizes,
        band_params,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_dataset,
    )

    xr.testing.assert_equal(computed_xr, expected_xr)


@parametrize_with_cases(
    "upper_left, lower_right, sizes, band_params_candidate, band_params_benchmark, nodata_value, encoded_nodata_value, shapes, band_dim_name, return_dataset, expected_candidate_xr, expected_benchmark_xr",
    glob="create_xarray_pairs",
)
def test_create_xarray_pairs(
    upper_left,
    lower_right,
    sizes,
    band_params_candidate,
    band_params_benchmark,
    nodata_value,
    encoded_nodata_value,
    shapes,
    band_dim_name,
    return_dataset,
    expected_candidate_xr,
    expected_benchmark_xr,
):
    """tests creating xarray from scratch"""

    computed_candidate_xr, computed_benchmark_xr = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_dataset,
    )

    xr.testing.assert_equal(computed_candidate_xr, expected_candidate_xr)
    xr.testing.assert_equal(computed_benchmark_xr, expected_benchmark_xr)


@parametrize_with_cases(
    "upper_left, lower_right, sizes, band_params, nodata_value, encoded_nodata_value, shapes, band_dim_name, return_dataset, expected_xr",
    glob="create_xarray_unsupported_shape_fail",
)
def test_create_xarray_unsupported_shape_fail(
    upper_left,
    lower_right,
    sizes,
    band_params,
    nodata_value,
    encoded_nodata_value,
    shapes,
    band_dim_name,
    return_dataset,
    expected_xr,
):
    """tests creating xarray from scratch"""

    with raises(ValueError):
        _create_xarray(
            upper_left,
            lower_right,
            sizes,
            band_params,
            nodata_value,
            encoded_nodata_value,
            shapes,
            band_dim_name,
            return_dataset,
        )
