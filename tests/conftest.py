"""
Configuration file for pytests
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Union  # , Optional

import os

import numpy as np
import xarray as xr

# import boto3

from gval.utils.loading_datasets import load_raster_as_xarray

# from config import PROJECT_DIR


# generate test data dir path
# TEST_DATA_DIR = os.path.join(PROJECT_DIR, "data", "data")


# name of S3 for test data
TEST_DATA_S3_NAME = "gval-test"
TEST_DATA_DIR = f"s3://{TEST_DATA_S3_NAME}"
# client
# TEST_DATA_S3_CLIENT = boto3.client("s3")

# def _check_file(
#     file_path: Union[str, os.PathLike],
#     test_data_s3_name: Optional[str] = TEST_DATA_S3_NAME,
#     test_data_s3_client: Optional[
#         boto3.session.botocore.client.BaseClient
#     ] = TEST_DATA_S3_CLIENT,
# ) -> None:
#     """
#     Downloads file if not already available locally.
#
#     TODO: Check for modified dates and only download if file has been updated on S3 end.
#
#     Parameters
#     ----------
#     file_path : Union[str, os.PathLike]
#         Local absolute file path the check.
#     test_data_s3_name : str, default = TEST_DATA_S3_NAME
#         Name of S3 to retrieve file from.
#     test_data_s3_client : boto3.session.botocore.client.BaseClient, default = TEST_DATA_S3_CLIENT
#         S3 client object from boto3.
#     """
#     # gets base name of file
#     file_name = os.path.basename(file_path)
#
#     # downloads file if not locally available
#     if not os.path.exists(file_path):
#         test_data_s3_client.download_file(test_data_s3_name, file_name, file_path)


"""
TODO:
 - Is there a way to create a stack of functions from _load_xarray, _build_map_file_path, and check_file that operates as a fixture/parameterization instead of normal python?
 - This would create less boilerplate code to maintain by avoiding calls to load_raster_as_xarray and _build_map_file_path
 - There is also no way to parameterize load_raster_as_xarray function.
"""


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


def _load_xarray(
    file_name: Union[str, os.PathLike],
    *args, **kwargs
    #masked: bool = False,
    #mask_and_scale: bool = False,
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
    return load_raster_as_xarray(file_path, *args, **kwargs)


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

    np.testing.assert_equal(list(computed_dict.keys()), list(expected_dict.keys()))
    np.testing.assert_equal(list(computed_dict.values()), list(expected_dict.values()))
