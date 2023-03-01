import os
import boto3
import pytest

from gval.utils.loading_datasets import load_raster_as_xarray
from config import TEST_DATA

test_data_dir = TEST_DATA


def check_file(file_path: str) -> str:
    """

    Parameters
    ----------
    file_path : str
        Name of file to check existence for

    Returns
    -------
    Full file path of object
    """

    file = file_path.split("/")[-1]
    if not os.path.exists(file_path):
        s3 = boto3.client("s3")
        s3.download_file("gval-test", file, file_path)

    return file_path


@pytest.fixture(scope="session", params=range(1))
def candidate_map_fp(request):
    """returns candidate maps"""
    filepath = check_file(
        os.path.join(test_data_dir, f"candidate_map_{request.param}.tif")
    )
    yield filepath


@pytest.fixture(scope="session")
def candidate_map(candidate_map_fp):
    """returns candidate map data"""
    yield load_raster_as_xarray(candidate_map_fp)


@pytest.fixture(scope="session", params=range(1))
def benchmark_map_fp(request):
    """returns benchmark maps"""
    filepath = check_file(
        os.path.join(test_data_dir, f"benchmark_map_{request.param}.tif")
    )
    yield filepath


@pytest.fixture(scope="session")
def benchmark_map(benchmark_map_fp):
    """returns benchmark map data"""
    yield load_raster_as_xarray(benchmark_map_fp)


@pytest.fixture(scope="session")
def agreement_map_fp(comparison):
    """return agreement maps"""
    _, agreement_map_key = comparison
    filepath = check_file(
        os.path.join(test_data_dir, f"agreement_map_{agreement_map_key}.tif")
    )
    yield filepath


@pytest.fixture(scope="session")
def agreement_map(agreement_map_fp):
    """return agreement map data"""
    yield load_raster_as_xarray(agreement_map_fp)
