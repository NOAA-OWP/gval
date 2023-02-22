"""
Comparison functionality
 - Includes pairing and comparison functions
 - crosstabbing functionality
 - Would np.meshgrid lend itself to pairing problem?
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Union,
)
from numbers import Number

import numpy as np
import pandas as pd
from xrspatial.zonal import crosstab
import xarray as xr


def _are_not_natural_numbers(c: Number, b: Number) -> None:
    """Checks pair to see if they are both natural numbers or two non-negative integers [0, 1, 2, 3, 4, ...)"""
    for x in (c, b):
        if np.isnan(x):  # checks to make sure it's not a nan value
            continue
        elif (x < 0) | ((x - int(x)) != 0):  # checks for non-negative and whole number
            raise ValueError(
                f"{x} is not natural numbers (non-negative integers) [0, 1, 2, 3, 4, ...)"
            )


## cantor method
def cantor_pair(c: int, b: int) -> int:
    """
    Produces unique natural number for two non-negative natural numbers (0,1,2,...)

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.

    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    _are_not_natural_numbers(c, b)
    return 0.5 * (c**2 + c + 2 * c * b + 3 * b + b**2)


def szudzik_pair(c: int, b: int) -> int:
    """
    Produces unique natural number for two non-negative natural numbers (0,1,2,3,...).

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.

    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    _are_not_natural_numbers(c, b)
    return c**2 + c + b if c >= b else b**2 + c


def negative_value_transformation(func: Callable):
    """
    Transforms negative values for use with pairing functions that only accept non-negative integers.

    Parameters
    ----------
    func : Callable
        Pairing function to apply negative value transformation to.

    Returns
    -------
    Callable
        Pairing function able to accept negative values.
    """
    _signing = lambda x: 2 * x if x >= 0 else -2 * x - 1

    def wrap(c, b):
        c = _signing(c)
        b = _signing(b)
        return func(c, b)

    return wrap


@negative_value_transformation
def cantor_pair_signed(c: int, b: int) -> int:
    """
    Output unique natural number for each unique combination of whole numbers using Cantor signed method.

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.

    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    return cantor_pair(c, b)


@negative_value_transformation
def szudzik_pair_signed(c: int, b: int) -> int:
    """
    Output unique natural number for each unique combination of whole numbers using Szudzik signed method._summary_

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.

    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    return szudzik_pair(c, b)


## user defined
def _make_pairing_dict(
    unique_candidate_values: Iterable, unique_benchmark_values: Iterable
):
    """Creates a dict pairing each unique value in candidate and benchmark arrays"""
    from itertools import product

    pairing_dict = {
        k: v
        for v, (k, _) in enumerate(
            product(unique_candidate_values, unique_benchmark_values)
        )
    }


def pairing_dict_fn(c: int, b: int) -> int:
    """
    Produces a pairing dictionary that produces a unique result for every combination ranging from 0 to the number of combinations.

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.

    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    return pairing_dict[c, b]


####################################
# compare


def compute_agreement_xarray(
    candidate: Union[xr.DataArray, xr.Dataset],
    benchmark: Union[xr.DataArray, xr.Dataset],
    comparison_function: Union[Callable, np.ufunc],
) -> Union[xr.DataArray, xr.Dataset]:
    # converts comparison function to ufunc with 2 inputs and 1 output if not already
    if not isinstance(comparison_function, np.ufunc):
        comparison_function_ufunc = np.frompyfunc(comparison_function, 2, 1)

    # use xarray apply ufunc to apply comparison to candidate and benchmark xarray's
    agreement_map_xr = xr.apply_ufunc((candidate, benchmark))

    return agreement_map_xr


def crosstab_xarray(
    candidate_map: xr.DataArray,
    benchmark_map: xr.DataArray,
    agreement_map: xr.DataArray,
    allow_list_candidate: Optional[Iterable[Union[int, float]]] = None,
    allow_list_benchmark: Optional[Iterable[Union[int, float]]] = None,
    exclude_values: Optional[Union[int, float]] = None,
) -> pd.DataFrame:
    """
    Crosstab xarray DataArrays to produce crosstab df.

    Parameters
    ----------
    candidate_map : xr.DataArray
        Candidate map
    benchmark_map : xr.DataArray
        Benchmark map
    agreement_map : xr.DataArray
        Agreement map
    allow_list_candidate : Optional[Iterable[Union[int,float]]], optional
        List of values in candidate to include in crosstab. Remaining values are excluded, by default None
    allow_list_benchmark : Optional[Iterable[Union[int,float]]], optional
        List of values in benchmark to include in crosstab. Remaining values are excluded, by default None
    exclude_values : Optional[Union[int,float]], optional
        List of values to exclude from crosstab, by default None

    Returns
    -------
    xr.DataArray
        Crosstab table with counts of each combination of candidate map and benchmark map values.
    """

    # according to xr-spatial docs:
    # Nodata value in values raster. Cells with nodata do not belong to any zone, and thus excluded from calculation.
    # would this be necessary? Does this just provide another means to exclude??

    # possible to use `xr.rio._obj.dims` or `xr.rio.check_dimensions()` to get first dimension name?
    # may not always be band?
    # See https://github.com/corteva/rioxarray/blob/9d5975624fa93b76c451457a97b342ba37dfc792/rioxarray/rioxarray.py
    # See https://github.com/corteva/rioxarray/blob/9d5975624fa93b76c451457a97b342ba37dfc792/rioxarray/raster_array.py

    # check dimensionality
    assert (
        candidate_map.shape == benchmark_map.shape
    ), f"Dimensionalities of candidate {candidate_map.shape} and benchmark {benchmark_map.shape} must match."

    # get extra dimension name
    extra_dim_name_candidate = candidate_map.rio._check_dimensions()
    extra_dim_name_benchmark = benchmark_map.rio._check_dimensions()

    # get length of extra dimension
    extra_dim_length = candidate_map[extra_dim_name_candidate].size

    # cycle through bands
    for b in range(1, extra_dim_length + 1):
        crosstab_df = crosstab(
            zones=candidate_map.sel({extra_dim_name_candidate: b}),
            values=benchmark_map.sel({extra_dim_name_benchmark: b}),
            zone_ids=allow_list_candidate,
            cat_ids=allow_list_benchmark,
            nodata_values=exclude_values,
        )

    """
    # try alternative approaches:
    - numpy histogram
       - np.histogram2d() (use with xr.apply_ufuncs?)
       - np.histogramdd() (use with xr.apply_ufuncs?)
       - np.histogram_bin_edges() (governs binning behaviour)
    - xhistogram:
       - xhistogram.xarray.histogram() (for xarray's?)
       - xhistogram.core.histogram() (for arrays?)
    - a function that inverses the pairing func to derive the source pairings and their respective counts.
        - make a ufunc that takes agreement array and produces three equal length arrays:
            - count for each agreement value
            - candidate value corresponding to each count
            - benchmark value corresponding to each count
    - try making a ufunc that takes in candidate and benchmark arrays and produces four equal length arrays:
        - unique values in candidate
        - unique values in benchmark
        - agreement values 
        - counts of agreement values
    """

    return crosstab_df
