"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest_cases import parametrize_with_cases
import xarray as xr

from gval.utils.loading_datasets import load_raster_as_xarray


@parametrize_with_cases("file_name")
def test_load_raster_as_xarray(file_name):
    """tests loading rasters as xarrays"""
    # NOTE: Do we need cases to test additional arguments if it is just wrapping rxr.open_rasterio()
    xr_map = load_raster_as_xarray(file_name)
    assert isinstance(
        xr_map, (xr.DataArray, xr.Dataset)
    ), "Did not load as xr.DataArray or xr.Dataset"
