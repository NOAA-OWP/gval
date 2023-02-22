"""
Comparison functionality
 - Includes pairing and comparison functions
 - crosstabbing functionality
 - Would np.meshgrid lend itself to pairing problem?
TODO:
    - Have not tested parallel case.
    - How to handle xr.Datasets, multiple bands, and multiple variables.
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import (
    Iterable,
    Optional,
    Union,
)
from numbers import Number

import numpy as np
import pandas as pd
from xrspatial.zonal import crosstab
import xarray as xr
import numba as nb


@nb.vectorize(nopython=True)
def _is_not_natural_number(x: Number) -> int:
    """
    Checks value to see if it is a natural number or two non-negative integer [0, 1, 2, 3, 4, ...)

    FIXME: Must return boolean or some other numba type. Having trouble returning none.
    """
    # checks to make sure it's not a nan value
    if np.isnan(x):
        return -2  # dummy return
    # checks for non-negative and whole number
    elif (x < 0) | ((x - int(x)) != 0):
        # FIXME: how to print x with message below using numba????
        raise ValueError(
            "Negative or non-whole number found (non-negative integers) [0, 1, 2, 3, 4, ...)"
        )
    # must return something according to signature
    else:
        return -2  # dummy return


@nb.vectorize(nopython=True)
def cantor_pair(c: Number, b: Number) -> Number:
    """
    Produces unique natural number for two non-negative natural numbers (0,1,2,...)

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.

    Returns
    -------
    Number
        Agreement map value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    _is_not_natural_number(c)
    _is_not_natural_number(b)
    return 0.5 * (c**2 + c + 2 * c * b + 3 * b + b**2)


@nb.vectorize(nopython=True)
def szudzik_pair(c: Number, b: Number) -> Number:
    """
    Produces unique natural number for two non-negative natural numbers (0,1,2,3,...).

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.

    Returns
    -------
    Number
        Agreement map value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    _is_not_natural_number(c)
    _is_not_natural_number(b)
    return c**2 + c + b if c >= b else b**2 + c


@nb.vectorize(nopython=True)
def _negative_value_transformation(x: Number) -> Number:
    """
    Transforms negative values for use with pairing functions that only accept non-negative integers.

    Parameters
    ----------
    x : Number
        Negative number to be transformed.

    Returns
    -------
    Number
        Transformed value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    return 2 * x if x >= 0 else -2 * x - 1


@nb.vectorize(nopython=True)
def cantor_pair_signed(c: Number, b: Number) -> Number:
    """
    Output unique natural number for each unique combination of whole numbers using Cantor signed method.

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.

    Returns
    -------
    Number
        Agreement map value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    ct = _negative_value_transformation(c)
    bt = _negative_value_transformation(b)
    return cantor_pair(ct, bt)


@nb.vectorize(nopython=True)
def szudzik_pair_signed(c: Number, b: Number) -> Number:
    """
    Output unique natural number for each unique combination of whole numbers using Szudzik signed method._summary_

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.

    Returns
    -------
    Number
        Agreement map value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    ct = _negative_value_transformation(c)
    bt = _negative_value_transformation(b)
    return szudzik_pair(ct, bt)


####################################
# compare


def compute_agreement_xarray(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    comparison_function: nb.vectorize,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Computes agreement map as xarray from candidate and benchmark xarray's.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.
    comparison_function : nb.np.ufunc.dufunc.DUFunc
        Numba vectorized comparison function. Created by decorating function with @nb.vectorize().

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Agreement map.

    References
    ----------
    .. [1] [Creating NumPy universal function](https://numba.readthedocs.io/en/stable/user/vectorize.html)
    """

    # use xarray apply ufunc to apply comparison to candidate and benchmark xarray's
    agreement_map = xr.apply_ufunc(
        comparison_function, candidate_map, benchmark_map, dask="forbidden"
    )

    """ TODO: What does "parallelized" option do?
        - If parallelized is selected, several other args should be considered
        - Including dask_gufunc_kwargs, output_dtypes, output_sizes, and meta. """

    return agreement_map


def _reorganize_crosstab_output(crosstab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorganizes crosstab output away from conventions used in xarray-spatial.

    Parameters
    ----------
    crosstab_df : pd.DataFrame
        Output of xarray-spatial.zonal.crosstab.

    Returns
    -------
    pd.DataFrame
        Crosstab dataframe using candidate and benchmark conventions.
    """

    crosstab_df = crosstab_df.set_index("zone").rename_axis(
        index="candidate", columns="benchmark"
    )

    return crosstab_df


def crosstab_xarray(
    candidate_map: xr.DataArray,
    benchmark_map: xr.DataArray,
    allow_list_candidate: Optional[Iterable[Number]] = None,
    allow_list_benchmark: Optional[Iterable[Number]] = None,
    exclude_value: Optional[Number] = None,
) -> pd.DataFrame:
    """
    Crosstab singular band xarray.DataArrays to produce crosstab df.

    TODO: Manage xr.DataArray's with multiple bands and xr.Datasets with variables of multiple bands.

    Parameters
    ----------
    candidate_map : xr.DataArray
        Candidate map with only one band.
    benchmark_map : xr.DataArray
        Benchmark map with only one band.
    allow_list_candidate : Optional[Iterable[Union[int,float]]], optional
        List of values in candidate to include in crosstab. Remaining values are excluded, by default None
    allow_list_benchmark : Optional[Iterable[Union[int,float]]], optional
        List of values in benchmark to include in crosstab. Remaining values are excluded, by default None
    exclude_value : Optional[Number], optional
        Value to exclude from crosstab, by default None

    Returns
    -------
    pd.DataFrame
        Crosstab dataframe with counts of each combination of candidate map and benchmark map values. Row index of dataframe represents unique values in candidate while columns represent unique values in benchmark.

    References
    ----------
    .. [1] [xarray.rio._check_dimensions()](https://github.com/corteva/rioxarray/blob/9d5975624fa93b76c451457a97b342ba37dfc792/rioxarray/rioxarray.py)
    .. [2] [xr.rio._obj.dims](https://github.com/corteva/rioxarray/blob/9d5975624fa93b76c451457a97b342ba37dfc792/rioxarray/raster_array.py)
    """

    # check dimensionality
    assert (
        candidate_map.shape == benchmark_map.shape
    ), f"Dimensionalities of candidate {candidate_map.shape} and benchmark {benchmark_map.shape} must match."

    """
    TODO:
    Use of `xr.rio._obj.dims` or `xr.rio._check_dimensions()` to get extra dimension name (sometimes called band)
        - is it always called band? may not always be band?
        - are there any other methods for doing this??
        - should this go elsewhere as it might be repeated?
    """

    # get extra dimension name
    extra_dim_name_candidate = candidate_map.rio._check_dimensions()
    extra_dim_name_benchmark = benchmark_map.rio._check_dimensions()

    # get length of extra dimension
    extra_dim_length = candidate_map[extra_dim_name_candidate].size

    # TEMP TODO: Support multi-band xarray's
    assert (
        extra_dim_length == 1
    ), "Crosstabbing is currently only supporting single band xr.DataArray's"

    # cycle through bands
    for b in range(1, extra_dim_length + 1):
        """
        NOTE: Consider that NoData is being tabulated here if it's not masked out.
        """
        crosstab_df = crosstab(
            zones=candidate_map.sel({extra_dim_name_candidate: b}),
            values=benchmark_map.sel({extra_dim_name_benchmark: b}),
            zone_ids=allow_list_candidate,
            cat_ids=allow_list_benchmark,
            nodata_values=exclude_value,
        )

        # reorganize df to follow candidate and benchmark conventions instead of xarray-spatial conventions
        crosstab_df = _reorganize_crosstab_output(crosstab_df)

    return crosstab_df
