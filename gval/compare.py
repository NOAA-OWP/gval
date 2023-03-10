"""
Comparison functionality
 - Includes pairing and comparison functions
 - crosstabbing functionality
 - Would np.meshgrid lend itself to pairing problem?
TODO:
    - Have not tested parallel case.
    - How to handle xr.Datasets, multiple bands, and multiple variables.
    - consider making a function registry to store pairing functions and their names in a dictionary
        - [Guide to function registration in Python with decorators](https://blog.miguelgrinberg.com/post/the-ultimate-guide-to-python-decorators-part-i-function-registration)
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Iterable, Optional, Union, Tuple
from numbers import Number

import numpy as np
import pandas as pd
from xrspatial.zonal import crosstab
import xarray as xr
import numba as nb


@nb.vectorize(nopython=True)
def _is_not_natural_number(x: Number) -> int:  # pragma: no cover
    """
    Checks value to see if it is a natural number or two non-negative integer [0, 1, 2, 3, 4, ...)

    Parameters
    ----------
    x : Number
        Number to test.

    Returns
    -------
    int
        Return -2 by default. Issue with numba usage. Please ignore for now.
        FIXME: Must return boolean or some other numba type. Having trouble returning none.

    Raises
    ------
    ValueError
        If x not a natural number.
    """
    # checks to make sure it's not a nan value
    if np.isnan(x):
        return -2  # dummy return
    # checks for non-negative and whole number
    elif (x < 0) | ((x - nb.int64(x)) != 0):
        # FIXME: how to print x with message below using numba????
        raise ValueError(
            "Non natural number found (non-negative integers, excluding Inf) [0, 1, 2, 3, 4, ...)"
        )
    # must return something according to signature
    else:
        return -2  # dummy return


@nb.vectorize(nopython=True)
def cantor_pair(c: Number, b: Number) -> Number:  # pragma: no cover
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
def szudzik_pair(c: Number, b: Number) -> Number:  # pragma: no cover
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
def _negative_value_transformation(x: Number) -> Number:  # pragma: no cover
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
def cantor_pair_signed(c: Number, b: Number) -> Number:  # pragma: no cover
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
def szudzik_pair_signed(c: Number, b: Number) -> Number:  # pragma: no cover
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


def _convert_dict_to_numba(
    py_dict: dict[Tuple[float, float], float]
) -> nb.typed.Dict[Tuple[nb.float64, nb.float64], nb.float64]:
    """
    Converts Python dict object to numba dict.

    NOTE: This is not currently being implemented.

    Parameters
    ----------
    py_dict : dict[Tuple[float, float], float]
        Python dictionary to convert to numba Dict.

    Returns
    -------
    nb.typed.Dict[nb.types.containers.UniTuple[nb.float64, nb.float64], nb.float64]
        Numba Dict.
    """

    # initiate numba dict
    # TODO: allow for multiple datatypes beyond float64 (float32, int, etc)
    nb_dict = nb.typed.Dict.empty(
        key_type=nb.types.containers.UniTuple(nb.float64, 2), value_type=nb.float64
    )

    for k, v in py_dict.items():
        nb_dict[k] = v

    return nb_dict


def _make_pairing_dict(
    unique_candidate_values: Iterable, unique_benchmark_values: Iterable
) -> dict[Tuple[Number, Number], Number]:
    """
    Creates a dict pairing each unique value in candidate and benchmark arrays.

    Parameters
    ----------
    unique_candidate_values : Iterable
        Unique values in candidate map to create pairing dict with.
    unique_benchmark_values : Iterable
        Unique values in benchmark map to create pairing dict with.

    Returns
    -------
    dict[Tuple[Number, Number], Number]
        Dictionary with keys consisting of unique pairings of candidate and benchmark values with value of agreement map for given pairing.
    """
    from itertools import product

    pairing_dict = {
        k: v
        for v, k in enumerate(product(unique_candidate_values, unique_benchmark_values))
    }

    return pairing_dict


@np.vectorize
def pairing_dict_fn(
    c: Number,
    b: Number,
    pairing_dict: dict[Tuple[Number, Number], Number],  # pragma: no cover
) -> Number:
    """
    Produces a pairing dictionary that produces a unique result for every combination ranging from 256 to the number of combinations.

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.
    pairing_dict : dict[Tuple[Number, Number], Number]
        Dictionary with keys of tuple with (c,b) and value to map agreement value to.

    Returns
    -------
    Number
        Agreement map value.
    """
    return pairing_dict[(c, b)]


####################################
# compare


def compute_agreement_xarray(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    comparison_function: Union[nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str],
    pairing_dict: dict = None,
    allow_candidate_values: Optional[Iterable[Number]] = None,
    allow_benchmark_values: Optional[Iterable[Number]] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Computes agreement map as xarray from candidate and benchmark xarray's.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.
    comparison_function : Union[nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str]
        Comparison function. Created by decorating function with @nb.vectorize() or using np.ufunc(). Use of numba is preferred as it is faster. Strings with registered comparison_functions are also accepted. Possible options include "pairing_dict". If passing "pairing_dict" value, please see the description for the argument for more information on behaviour.
    pairing_dict: dict[Tuple[Number, Number] : Number], default = None
        When "pairing_dict" is used for the comparison_function argument, a pairing dictionary can be passed by user. A pairing dictionary is structured as `{(c, b) : a}` where `(c, b)` is a tuple of the candidate and benchmark value pairing, respectively, and `a` is the value for the agreement array to be used for this pairing.

        If None is passed for pairing_dict, the allow_candidate_values and allow_benchmark_values arguments are required. For this case, the pairings in these two iterables will be paired in the order provided and an agreement value will be assigned to each pairing starting with 0 and ending with the number of possible pairings.

        A pairing dictionary can be used by the user to note which values to allow and which to ignore for comparisons. It can also be used to decide how nans are handled for cases where either the candidate and benchmark maps have nans or both.
    allow_candidate_values : Optional[Iterable[Union[int,float]]], default = None
        List of values in candidate to include in computation of agreement map. Remaining values are excluded. If "pairing_dict" is set selected for comparison_function and pairing_function is None, this argument is necessary to construct the dictionary. Otherwise, this argument is optional and by default this value is set to None and all values are considered.
    allow_benchmark_values : Optional[Iterable[Union[int,float]]], default = None
        List of values in benchmark to include in computation of agreement map. Remaining values are excluded. If "pairing_dict" is set selected for comparison_function and pairing_function is None, this argument is necessary to construct the dictionary. Otherwise, this argument is optional and by default this value is set to None and all values are considered.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Agreement map.

    References
    ----------
    .. [1] [Creating NumPy universal function](https://numba.readthedocs.io/en/stable/user/vectorize.html)
    .. [2] [NumPy vectorize](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html)
    .. [3] [NumPy frompyfunc](https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html)
    """

    """
    TODO:
    What does dask argument in xr.apply_ufunc do?
        - If parallelized is selected, several other args should be considered
        - Including dask_gufunc_kwargs, output_dtypes, output_sizes, and meta.
    nan management is still not clear across the cases.
        - masking currently turns everything to nan.
        - How to handle this??
    """

    # sets dask argument for xr.apply_ufunc
    dask_status = "parallelized"

    ############################################################################################
    # Masking out values not in allowed lists

    # mask out values not in allow_candidate_values
    if allow_candidate_values is not None:
        candidate_map = candidate_map.where(candidate_map.isin(allow_candidate_values))

    # mask out values not in allow_benchmark_values
    if allow_benchmark_values is not None:
        benchmark_map = benchmark_map.where(benchmark_map.isin(allow_benchmark_values))

    ###########################################################################################
    # Handling for pairing_dict functionality

    if comparison_function == "pairing_dict":  # when pairing_dict is a dict
        if pairing_dict is None:  # this is used for when pairing_dict is not passed
            # user must set arguments to build pairing dict, throws value error
            # TODO: consider allow use of unique to acquire all values from candidate and benchmarks
            if (allow_candidate_values is None) | (allow_benchmark_values is None):
                raise ValueError(
                    "When comparison_function argument is set to 'pairing_dict', must pass values for allow_candidate_values and allow_benchmark_values arguments."
                )

            # this creates the pairing dictionary from the passed allowed values
            pairing_dict = _make_pairing_dict(
                allow_candidate_values, allow_benchmark_values
            )

        """
        FIXME:
        When pairing_dict_fn is decorated with @nb.vectorize(nopython=True). A typing error occurs.
            - Noticed that np.vectorize seems to perform well so removing numba for pairing_dict
            - Line of code to use if using numba:
                - pairing_dict = _convert_dict_to_numba(pairing_dict)
        """

        print(candidate_map.dtype)

        # this return is for the pairing_dict case
        return xr.apply_ufunc(
            pairing_dict_fn,
            candidate_map,
            benchmark_map,
            [
                pairing_dict
            ],  # encapsulating this in a list is necessary for vectorization
            dask=dask_status,
        )

    ###########################################################################################
    # for cases when pairing dictionaries are not being used at all.

    """ TODO: Can we handle vectorizing functions here for the user if needed?
        - use nb.vectorize, np.vectorize, np.frompyfunc as needed, or set vectorize=True within xr.apply_ufunc
        - this could work if user's want to pass a normal python function.
    """

    # use xarray apply ufunc to apply comparison to candidate and benchmark xarray's
    return xr.apply_ufunc(
        comparison_function, candidate_map, benchmark_map, dask=dask_status
    )


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
    allow_candidate_values: Optional[Iterable[Number]] = None,
    allow_benchmark_values: Optional[Iterable[Number]] = None,
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
    allow_candidate_values : Optional[Iterable[Union[int,float]]], default = None
        List of values in candidate to include in crosstab. Remaining values are excluded.
    allow_benchmark_values : Optional[Iterable[Union[int,float]]], default = None
        List of values in benchmark to include in crosstab. Remaining values are excluded.
    exclude_value : Optional[Number], default = None
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
    TODO:
    Consider cases with 1) multi-band DataArray, 2) multi-variable Dataset, 3) multi-variable Dataset with multi-bands
        - Use [map](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.map.html#xarray.Dataset.map) for Dataset to apply function to every variable
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
            zone_ids=allow_candidate_values,
            cat_ids=allow_benchmark_values,
            nodata_values=exclude_value,
        )

        # reorganize df to follow candidate and benchmark conventions instead of xarray-spatial conventions
        crosstab_df = _reorganize_crosstab_output(crosstab_df)

    return crosstab_df
