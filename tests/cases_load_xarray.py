"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest_cases import parametrize

from tests.conftest import _build_map_file_path


file_names = [
    "candidate_map_0.tif",
    "benchmark_map_0.tif",
    "candidate_map_1.tif",
    "benchmark_map_1.tif",
]


@parametrize("file_name", file_names)
def case_map(file_name):
    """returns file path to map given a file_name"""
    return _build_map_file_path(file_name)
