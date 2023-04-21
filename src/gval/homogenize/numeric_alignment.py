# numeric_alignment.py
"""
Functions to check for and ensure numeric datatypes are compatible in numeric processing

"""

from typing import Union, Tuple

import numpy as np
import xarray as xr


def _align_numeric_dtype(
    candidate_map: xr.DataArray, benchmark_map: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Align numeric datatypes to the highest precision to avoid warnings and possible issues with numba
    vectorized operations and/or dask operations.

    Parameters
    ----------
    candidate_map: xr.DataArray
        Candidate map xarray DataArray
    benchmark_map: xr.DataArray
        Benchmark map xarray DataArray

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        Numerically aligned datatype candidate and benchmark maps

    """

    # Determine order for aligning numeric data type
    c_is_int, b_is_int = (
        np.issubdtype(candidate_map.dtype, np.integer),
        np.issubdtype(benchmark_map.dtype, np.integer),
    )
    total_c = candidate_map.dtype.alignment + int(not c_is_int) * 5
    total_b = benchmark_map.dtype.alignment + int(not b_is_int) * 5

    def change_numeric_type(original_map, target_map):
        original_map_data = original_map.data.astype(target_map.data.dtype)
        adjusted_map = original_map.copy(
            data=xr.where(
                original_map == original_map.rio.nodata,
                original_map.rio.nodata,
                original_map_data,
            )
        )
        adjusted_map = adjusted_map.rio.set_nodata(original_map.rio.nodata)
        adjusted_map.rio.write_nodata(original_map.rio.nodata, encoded=True)

        return adjusted_map

    # If candidate is higher precision
    if total_c > total_b:
        benchmark_map = change_numeric_type(
            original_map=benchmark_map, target_map=candidate_map
        )
    # If benchmark is higher precision
    if total_b > total_c:
        candidate_map = change_numeric_type(
            original_map=candidate_map, target_map=benchmark_map
        )

    return candidate_map, benchmark_map


def _align_datasets_dtype(
    candidate_map: xr.Dataset, benchmark_map: xr.Dataset
) -> xr.Dataset:
    """
    Iteration through data variables for two datasets to align numerical dtypes

    Parameters
    ----------
    candidate_map: xr.Dataset
        Candidate map xarray dataset
    benchmark_map: xr.Dataset
        Benchmark map xarray dataset

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]

    """

    for c_var, b_var in zip(candidate_map.data_vars, benchmark_map.data_vars):
        candidate_map[c_var], benchmark_map[b_var] = _align_numeric_dtype(
            candidate_map[c_var], benchmark_map[b_var]
        )

    return candidate_map, benchmark_map


def _align_numeric_data_type(
    candidate_map: Union[xr.Dataset, xr.DataArray],
    benchmark_map: Union[xr.Dataset, xr.DataArray],
) -> Tuple[Union[xr.Dataset, xr.DataArray], Union[xr.Dataset, xr.DataArray]]:
    """
    Align data type for xarray objects

    Parameters
    ----------
    candidate_map: Union[xr.Dataset, xr.DataArray]
        Candidate map xarray object
    benchmark_map: Union[xr.Dataset, xr.DataArray]
        Benchmark map xarray dataset

    Returns
    -------
    Tuple[Union[xr.Dataset, xr.DataArray], Union[xr.Dataset, xr.DataArray]]

    """

    if isinstance(candidate_map, xr.DataArray):
        return _align_numeric_dtype(candidate_map, benchmark_map)
    else:
        return _align_datasets_dtype(candidate_map, benchmark_map)
