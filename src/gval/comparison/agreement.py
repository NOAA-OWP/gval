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


from typing import Iterable, Optional, Union, Tuple, Callable, Dict
from numbers import Number
from itertools import product

import numpy as np
import xarray as xr
import numba as nb
import dask

from gval.comparison.pairing_functions import (
    _make_pairing_dict_fn,
)
from gval.utils.loading_datasets import _handle_xarray_memory, _check_dask_array


def _compute_agreement_map(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    comparison_function: Union[
        Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str
    ],
    pairing_dict: Optional[Dict[Tuple[Number, Number], Number]] = None,
    allow_candidate_values: Optional[Iterable[Number]] = None,
    allow_benchmark_values: Optional[Iterable[Number]] = None,
    nodata: Optional[Number] = None,
    encode_nodata: Optional[bool] = False,
    continuous: Optional[bool] = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Computes agreement map as xarray from candidate and benchmark xarray's.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.
    comparison_function : Union[Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str]
        Comparison function. Created by decorating function with @nb.vectorize() or using np.ufunc(). Use of numba is preferred as it is faster. Strings with registered comparison_functions are also accepted. Possible options include "pairing_dict". If passing "pairing_dict" value, please see the description for the argument for more information on behaviour.
    pairing_dict: Optional[Dict[Tuple[Number, Number], Number]], default = None
        When "pairing_dict" is used for the comparison_function argument, a pairing dictionary can be passed by user. A pairing dictionary is structured as `{(c, b) : a}` where `(c, b)` is a tuple of the candidate and benchmark value pairing, respectively, and `a` is the value for the agreement array to be used for this pairing.

        If None is passed for pairing_dict, the allow_candidate_values and allow_benchmark_values arguments are required. For this case, the pairings in these two iterables will be paired in the order provided and an agreement value will be assigned to each pairing starting with 0 and ending with the number of possible pairings.

        A pairing dictionary can be used by the user to note which values to allow and which to ignore for comparisons. It can also be used to decide how nans are handled for cases where either the candidate and benchmark maps have nans or both.
    allow_candidate_values : Optional[Iterable[Union[int,float]]], default = None
        List of values in candidate to include in computation of agreement map. Remaining values are excluded. If "pairing_dict" is set selected for comparison_function and pairing_function is None, this argument is necessary to construct the dictionary. Otherwise, this argument is optional and by default this value is set to None and all values are considered.
    allow_benchmark_values : Optional[Iterable[Union[int,float]]], default = None
        List of values in benchmark to include in computation of agreement map. Remaining values are excluded. If "pairing_dict" is set selected for comparison_function and pairing_function is None, this argument is necessary to construct the dictionary. Otherwise, this argument is optional and by default this value is set to None and all values are considered.
    nodata : Optional[Number], default = None
        No data value to write to agreement map output. This will use `rxr.rio.write_nodata(nodata)`.
    encode_nodata : Optional[bool], default = False
        Encoded no data value to write to agreement map output. A nodata argument must be passed. This will use `rxr.rio.write_nodata(nodata, encode=encode_nodata)`.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Agreement map.

    Raises
    ------
    ValueError
        Must pass a value for 'nodata' argument if setting 'encode_nodata' to True.

    References
    ----------
    .. [1] `Creating NumPy universal function <https://numba.readthedocs.io/en/stable/user/vectorize.html>`_
    .. [2] `NumPy vectorize <https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html>`_
    .. [3] `NumPy frompyfunc <https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html>`_
    .. [4] `Rioxarray Manage Information Loss <https://corteva.github.io/rioxarray/stable/getting_started/manage_information_loss.html>`_
    .. [5] `Rioxarray NoData Management <https://corteva.github.io/rioxarray/stable/getting_started/nodata_management.html>`_
    """

    """
    TODO:
    - Understand how these arguments affect the agreement map.
    - A partial list of some important arguments is included below with either default or presumed values.
    - These arguments need to be further researched and tested.
    - `keep_attrs` argument should probably be handled here as to be able to preserve both candidate and benchmark attributes. Possibly consider using a prefix or suffix for attribute keys to denote the source map of the attribute (e.g. {candidate_key: value, benchmark_key: value})
    - `dask` needs testing with dask arrays.
    - `dataset_join` needs testing when data variable names in candidate and benchmark differ
    - `output_dtypes` should be considered to manage dtypes for agreement map
    - behavior for agreement map should consider how behavior for attributes, variable names, and dtypes in handled within crosstabbing functionality.
    - read and consider all of the arguments in (xr.apply_ufunc documentation)[https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html]
    - if the attributes or dtypes of the output agreement map are important, the tests must consider these values by the using the correct `xr.testing.assert_*` function.
        - Consider switching test to `xr.testing.assert_identical()` to consider names and attributes.
        - If names are not to be tested, consider using `tests.conftest._assert_pairing_dict_equal()` for attribute testing.
    """

    # some input checking to avoid computing ValueErrors

    # nodata is None & encode_nodata is True
    if nodata is None:
        if encode_nodata:
            raise ValueError(
                "Must pass a value for 'nodata' argument if setting 'encode_nodata' to True."
            )

    # sets kwargs for xr.apply_ufunc
    apply_ufunc_kwargs = {
        "dask": "parallelized",  # how does this work on dask arrays?
        "keep_attrs": True,  # default, copies attrs from first input
        "join": "exact",  # default, raise ValueError instead of aligning when indexes to be aligned are not equal
        "dataset_join": "exact",  # default, data variables on all Dataset objects must match exactly
        "output_dtypes": None,  # default, Optional list of output dtypes. Only used if dask='parallelized' or vectorize=True.
    }

    ############################################################################################

    def _manage_information_loss(agreement_map, crs, nodata, encode_nodata, dtype):
        """
        Manages the information loss due to `xr.apply_ufunc`

        This encapsulated function is to manage information loss that can't be managed with apply_ufunc_kwargs.
        """

        # sets CRS that is lost with `xr.apply_ufunc`
        agreement_map.rio.write_crs(crs, inplace=True)

        # setting agreement map nodata and encoded nodata
        if nodata is not None:
            # this masks the desired nodata value within agreement
            if encode_nodata:
                agreement_map = agreement_map.astype(dtype).where(
                    agreement_map != nodata
                )

                # writes no data and encoded no data if set
                agreement_map.rio.write_nodata(
                    nodata, encoded=encode_nodata, inplace=True
                )
            else:
                agreement_map.rio.write_nodata(nodata, inplace=True)

        return agreement_map

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

    ufunc_args = [candidate_map, benchmark_map]

    if comparison_function == "pairing_dict":  # when pairing_dict is a dict
        # this creates the pairing dictionary from the passed allowed values
        comparison_function = _make_pairing_dict_fn(
            pairing_dict=pairing_dict,
            unique_candidate_values=allow_candidate_values,
            unique_benchmark_values=allow_benchmark_values,
        )

    # use xarray apply ufunc to apply comparison to candidate and benchmark xarray's
    # NOTE: Default behavior loses CRS and uses nodata from first argument (candidate_map)

    agreement_map = xr.apply_ufunc(
        comparison_function, *ufunc_args, **apply_ufunc_kwargs
    )

    is_dask = _check_dask_array(candidate_map)

    def get_unique_values(candidate, benchmark):
        unique_c = (
            dask.array.unique(candidate.data).compute()
            if is_dask
            else np.unique(candidate)
        )
        unique_b = (
            dask.array.unique(benchmark.data).compute()
            if is_dask
            else np.unique(benchmark)
        )

        return unique_c, unique_b

    # Add pairing dictionary and reverse pairing dictionary to agreement map attributes
    if pairing_dict is not None and not continuous:
        agreement_map.attrs["pairing_dictionary"] = pairing_dict

    if pairing_dict is None and not continuous:
        if allow_candidate_values is None or allow_benchmark_values is None:
            if isinstance(candidate_map, xr.Dataset):
                for idx, var in enumerate(candidate_map.data_vars):
                    agreement_map[var].attrs["pairing_dictionary"] = {
                        (x, y): comparison_function(x, y)
                        for x, y in product(
                            *get_unique_values(candidate_map[var], benchmark_map[var])
                        )
                    }

                    if idx == 0:
                        agreement_map.attrs

            else:
                agreement_map.attrs["pairing_dictionary"] = {
                    (x, y): comparison_function(x, y)
                    for x, y in product(
                        *get_unique_values(candidate_map, benchmark_map)
                    )
                }
        else:
            agreement_map.attrs["pairing_dictionary"] = {
                (x, y): comparison_function(x, y)
                for x, y in product(allow_candidate_values, allow_benchmark_values)
            }

    if isinstance(candidate_map, xr.DataArray):
        agreement_map = _manage_information_loss(
            agreement_map=agreement_map,
            crs=candidate_map.rio.crs,
            nodata=nodata,
            encode_nodata=encode_nodata,
            dtype=candidate_map.dtype,
        )
    else:
        for c_var, a_var in zip(candidate_map.data_vars, agreement_map.data_vars):
            agreement_map[a_var] = _manage_information_loss(
                agreement_map=agreement_map[a_var],
                crs=candidate_map[c_var].rio.crs,
                nodata=nodata,
                encode_nodata=encode_nodata,
                dtype=candidate_map[c_var].dtype,
            )

    return _handle_xarray_memory(agreement_map, make_temp=True)
