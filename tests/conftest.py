"""
Configuration file for pytests
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Union

import os

import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import pandas as pd


# name of S3 for test data
TEST_DATA_S3_NAME = "gval-test"
TEST_DATA_DIR = f"s3://{TEST_DATA_S3_NAME}"


def _build_map_file_path(file_name: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    """
    Returns local file path for a given file name.

    Retrieves file from S3 if not already available.

    Parameters
    ----------
    file_name : Union[str, os.PathLike]
        Base filename.

    Returns
    -------
    Union[str, os.PathLike]
        Local, absolute file path for input.
    """
    file_path = os.path.join(TEST_DATA_DIR, file_name)
    # _check_file(file_path)
    return file_path


def _load_gpkg(
    file_name: Union[str, os.PathLike],
    *args,
    **kwargs
    # masked: bool = False,
    # mask_and_scale: bool = False,
) -> gpd.GeoDataFrame:
    """
    Loads geopackage given a base file name.

    Parameters
    ----------
    file_name : Union[str, os.PathLike]
        Base file name of file within local TEST_DATA_DIR or TEST_DATA_S3_NAME.

    Returns
    -------
    gpd.GeoDataFrame
        geopandas GeoDataFrame.
    """
    file_path = _build_map_file_path(file_name)
    return gpd.read_file(file_path, *args, **kwargs)


def _load_xarray(
    file_name: Union[str, os.PathLike],
    *args,
    **kwargs
    # masked: bool = False,
    # mask_and_scale: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Loads xarray given a base file name.

    Parameters
    ----------
    file_name : Union[str, os.PathLike]
        Base file name of file within local TEST_DATA_DIR or TEST_DATA_S3_NAME.
    mask_and_scale: bool


    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        xarray object.
    """
    file_path = _build_map_file_path(file_name)
    return rxr.open_rasterio(file_path, *args, **kwargs)


def _assert_pairing_dict_equal(computed_dict: dict, expected_dict: dict) -> None:
    """
    Testing function used to test if two pairing dictionaries are equal.

    This is necessary because np.nans can be of float or np.float64 kind which makes operator (==) comparisons false.

    Parameters
    ----------
    computed_dict : dict
        Pairing dict computed to test.
    expected_dict : dict
        Expected pairing dict to compare to.

    Returns
    -------
    None

    See also
    --------
    :obj:`np.testing.assert_equal`

    Raises
    ------
    AssertionError
    """

    try:
        np.testing.assert_allclose(
            list(computed_dict.keys()), list(expected_dict.keys())
        )
    except np.exceptions.DTypePromotionError:
        np.testing.assert_equal(
            list(computed_dict.keys()), list(expected_dict.keys())
        )
    
    try:
        np.testing.assert_allclose(
            list(computed_dict.values()), list(expected_dict.values())
        )
    except np.exceptions.DTypePromotionError:
        np.testing.assert_equal(
            list(computed_dict.values()), list(expected_dict.values())
        )

    # checks keys to make sure they hash they same way
    for k, v in computed_dict.items():
        expected_dict[k]

    # checks keys to make sure they hash they same way
    for k, v in expected_dict.items():
        computed_dict[k]


def _attributes_to_string(
    obj: Union[xr.DataArray, xr.Dataset]
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Converts attributes to string to mimic a raster loaded from disk

    Parameters
    ----------
    obj: Union[xr.DataArray, xr.Dataset]
        Object to convert properties in string format

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        object with properties converted to string format
    """
    if "pairing_dictionary" in obj.attrs and isinstance(
        obj.attrs["pairing_dictionary"], dict
    ):
        obj.attrs["pairing_dictionary"] = str(obj.attrs["pairing_dictionary"])

    return obj


def _compare_metrics_df_with_xarray(
    metrics_df: pd.DataFrame, expected_df: pd.DataFrame
):
    """
    Compares metrics dataframe with expected dataframe and raises AssertionError if they do not match.

    Used to compare dataframes with xarray objects.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe to compare.
    expected_df : pd.DataFrame
        Expected metrics dataframe to compare to.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
    """

    # compare indices and columns
    assert metrics_df.index.equals(
        expected_df.index
    ), "Metrics dataframe indices do not match expected indices. "
    assert metrics_df.columns.equals(
        expected_df.columns
    ), "Metrics dataframe columns do not match expected columns. "

    for r in range(metrics_df.shape[0]):
        for c in range(metrics_df.shape[1]):
            m = metrics_df.iloc[r, c]
            e = expected_df.iloc[r, c]

            if isinstance(m, (xr.DataArray, xr.Dataset)):
                xr.testing.assert_allclose(m, e, atol=1e-8)
            elif isinstance(m, Exception):
                assert isinstance(m, e)
            else:
                assert m == e
