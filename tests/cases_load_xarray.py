"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np
import xarray as xr
from pytest_cases import parametrize

from tests.conftest import _build_map_file_path


file_names = [
    "candidate_map_0.tif",
    "benchmark_map_0.tif",
    "candidate_map_1.tif",
    "benchmark_map_1.tif",
    "target_map_0.tif",
    "target_map_1.tif",
]


@parametrize("file_name", file_names)
def case_map(file_name):
    """returns file path to map given a file_name"""
    return _build_map_file_path(file_name)


def case_create_xarray_array_success():

    upper_left = (0, 0)
    lower_right = (10, 10)
    sizes = (10, 10)
    nodata_value = np.nan
    encoded_nodata_value = -1
    shapes = "circle"
    band_dim_name = "band"
    return_dataset = False

    
    # create xarray
    # background value, circle value, circle center, and circle radius
    band_params = [
        (0, 1, (7, 7), 3), # band 1
    ]

    data = np.array([[
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan, np.nan, np.nan, np.nan, np.nan,  1.,  1.,  1.,  1.,  1.]
    ]])

    expected_xr = xr.DataArray(
        data=data,
        dims=["band", "y", "x"],
        coords={
            "spatial_ref" : 0,
            "band" : [1],
            "y": np.linspace(upper_left[1], lower_right[1], sizes[1]),
            "x": np.linspace(lower_right[0], upper_left[0], sizes[0])
        },
    )

    return (
        upper_left, lower_right, sizes, band_params, nodata_value, encoded_nodata_value, shapes, band_dim_name, return_dataset, expected_xr
    )

def case_create_xarray_dataset_success():

    upper_left = (0, 0)
    lower_right = (10, 10)
    sizes = (10, 10)
    nodata_value = np.nan
    encoded_nodata_value = -1
    shapes = "circle"
    band_dim_name = "band"
    return_dataset = True

    # create xarray
    # background value, circle value, circle center, and circle radius
    band_params = [
        (0, 1, (7, 7), 3), # band 1
    ]

    data = np.array([[
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan, np.nan, np.nan, np.nan, np.nan, 1.,  1.,  1.,  1.,  1.]
    ]])

    expected_xr = xr.DataArray(
        data=data,
        dims=["band", "y", "x"],
        coords={
            "spatial_ref" : 0,
            "band" : [1],
            "y": np.linspace(upper_left[1], lower_right[1], sizes[1]),
            "x": np.linspace(lower_right[0], upper_left[0], sizes[0])
        },
    ).to_dataset(name="variable")

    return (
        upper_left, lower_right, sizes, band_params, nodata_value, encoded_nodata_value, shapes, band_dim_name, return_dataset, expected_xr
    )

def case_create_xarray_pairs():

    upper_left = (0, 0)
    lower_right = (10, 10)
    size = 10
    nodata_value = np.nan
    encoded_nodata_value = -1
    shapes = "circle"
    band_dim_name = "band"
    return_dataset = False
    
    # create xarray
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 1, (7, 7), 3), # band 1
        (0, 2, (7, 7), 3), # band 2
    ]

    band_params_benchmark = [
        (0, 2, (7, 7), 3), # band 1
        (0, 1, (7, 7), 3), # band 2
    ]

    data_band_1 = np.array([[
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan, np.nan, np.nan, np.nan, np.nan,  1.,  1.,  1.,  1.,  1.]
    ]])

    data_band_2 = data_band_1.copy()
    data_band_2[data_band_1 == 1] = 2

    data = np.concatenate([data_band_1, data_band_2], axis=0)

    expected_candidate_xr = xr.DataArray(
        data=data,
        dims=["band", "y", "x"],
        coords={
            "spatial_ref" : 0,
            "band" : [1, 2],
            "y": np.linspace(upper_left[1], lower_right[1], size),
            "x": np.linspace(lower_right[0], upper_left[0], size)
        },
    )

    data = np.concatenate([data_band_2, data_band_1], axis=0)

    expected_benchmark_xr = xr.DataArray(
        data=data,
        dims=["band", "y", "x"],
        coords={
            "spatial_ref" : 0,
            "band" : [1, 2],
            "y": np.linspace(upper_left[1], lower_right[1], size),
            "x": np.linspace(lower_right[0], upper_left[0], size)
        },
    )

    return (
        upper_left, lower_right, size, band_params_candidate, band_params_benchmark, nodata_value, encoded_nodata_value, shapes, band_dim_name, return_dataset, expected_candidate_xr, expected_benchmark_xr
    )


def case_create_xarray_unsupported_shape_fail():

    upper_left = (0, 0)
    lower_right = (10, 10)
    sizes = (10, 10)
    nodata_value = np.nan
    encoded_nodata_value = -1
    shapes = "square"
    band_dim_name = "band"
    return_dataset = False

    
    # create xarray
    # background value, circle value, circle center, and circle radius
    band_params = [
        (0, 1, (7, 7), 3), # band 1
    ]

    data = np.array([[
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., np.nan],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
        [np.nan,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
        [np.nan, np.nan, np.nan, np.nan, np.nan, 1.,  1.,  1.,  1.,  1.]
    ]])

    expected_xr = xr.DataArray(
        data=data,
        dims=["band", "y", "x"],
        coords={
            "spatial_ref" : 0,
            "band" : [1],
            "y": np.linspace(upper_left[1], lower_right[1], sizes[1]),
            "x": np.linspace(lower_right[0], upper_left[0], sizes[0])
        },
    )

    return (
        upper_left, lower_right, sizes, band_params, nodata_value, encoded_nodata_value, shapes, band_dim_name, return_dataset, expected_xr
    )
