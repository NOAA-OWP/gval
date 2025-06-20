"""
Configuration file for pytests
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Union

import os

import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import pandas as pd
from deepdiff import DeepDiff

from gval.comparison.pairing_functions import PairingDict

# name of S3 for test data
TEST_DATA_S3_NAME = "gval"
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
    file_name: Union[str, os.PathLike], *args, engine: str = "pyogrio", **kwargs
) -> gpd.GeoDataFrame:
    """
    Loads geopackage given a base file name.

    Parameters
    ----------
    file_name : Union[str, os.PathLike]
        Base file name of file within local TEST_DATA_DIR or TEST_DATA_S3_NAME.
    engine : str, default = "pyogrio"
        Engine to use to read file. Accepts "fiona" or "pyogrio".

    Returns
    -------
    gpd.GeoDataFrame
        geopandas GeoDataFrame.
    """
    file_path = _build_map_file_path(file_name)

    # pop engine from kwargs if it exists
    if "engine" in kwargs:
        engine = kwargs.pop("engine")

    return gpd.read_file(file_path, engine=engine, *args, **kwargs)


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


def _assert_pairing_dict_equal(
    computed_dict: Union[dict, PairingDict], expected_dict: Union[dict, PairingDict]
) -> None:
    """
    Testing function used to test if two pairing dictionaries are equal.

    Parameters
    ----------
    computed_dict : dict or PairingDict
        Pairing dict computed to test.
    expected_dict : dict or PairingDict
        Expected pairing dict to compare to.

    Returns
    -------
    None

    See also
    --------
    :obj:`deepdiff.DeepDiff`

    Raises
    ------
    AssertionError
    """
    # compute difference between dictionaries. If empty dict, they are equal
    diff = DeepDiff(expected_dict, computed_dict, significant_digits=5)

    if diff:
        raise AssertionError(f"Dictionaries are not equal. {diff}")


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
