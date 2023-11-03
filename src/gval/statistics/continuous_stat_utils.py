"""
Continuous Statistics Functions From Error Based Agreement Maps.
"""

from typing import Callable, Union, Optional
from functools import wraps

import xarray as xr

from gval.utils.loading_datasets import _check_dask_array


def convert_output(func: Callable) -> Callable:  # pragma: no cover
    """
    Decorator that converts the output of a function to a different type.

    If the output is an xarray.DataArray, it is converted to a single numeric value.
    If the output is an xarray.Dataset, it is converted to a dictionary with the band names as keys.

    Parameters
    ----------
    func : Callable
        The function whose output is to be converted.

    Returns
    -------
    wrapper : Callable
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call the decorated function
        result = func(*args, **kwargs)

        is_dsk = _check_dask_array(result)
        if isinstance(result, xr.DataArray):
            # Convert to a single numeric value
            return result.compute().item() if is_dsk else result.item()
        elif isinstance(result, xr.Dataset):
            # Convert to a dictionary with band names as keys
            if is_dsk:
                return {
                    band: result[band].compute().item() for band in result.data_vars
                }
            else:
                return {band: result[band].item() for band in result.data_vars}

        return result

    return wrapper


def compute_error_if_none(func: Callable) -> Callable:  # pragma: no cover
    """
    Decorator to compute the error as the difference between the candidate map and benchmark map
    if the error is not supplied.

    The decorated function should have three positional arguments:
    error: Union[xr.DataArray, xr.Dataset], optional
        Candidate minus benchmark error.
    benchmark_map: Union[xr.DataArray, xr.Dataset], optional
        Benchmark map.
    candidate_map: Union[xr.DataArray, xr.Dataset], optional
        Candidate map.

    The error is computed as the difference between the candidate_map and the benchmark_map.
    If the error is provided, it is used directly.

    If none of the arguments are provided, a ValueError is raised.

    Parameters
    ----------
    func : Callable
        Function to be decorated.

    Returns
    -------
    wrapper : Callable
        The decorated function.
    """

    def wrapper(
        error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
        benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
        candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    ):
        if error is None:
            if benchmark_map is not None and candidate_map is not None:
                error = candidate_map - benchmark_map
            else:
                raise ValueError(
                    "Must provide either `error` or both `candidate_map` and `benchmark_map`."
                )
        return func(error, benchmark_map, candidate_map)

    return wrapper
