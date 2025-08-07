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

from typing import Union, Iterable
from numbers import Number

import numpy as np
import pandas as pd
import xarray as xr
import pandera.pandas as pa
from pandera.typing import DataFrame
import dask
from flox.xarray import xarray_reduce

from gval.utils.schemas import Crosstab_df
from gval.utils.loading_datasets import _check_dask_array


def _crosstab_docstring(dimension: Union[int, str], xarray_obj: str = "xr.DataArray"):
    """
    Docstring decorator for crosstab functions.

    Parameters
    ----------
    dimension : Union[int, str]
        Number of dimensions function will support. Use either 2, 3, or 2/3.
    xarray_obj : str, default = "xr.DataArray"
        Type of xarray object function accepts. xr.DataArray or xr.Dataset.

    Returns
    -------
    Callable
        Decorated crosstab_* function with new docstring.

    Raises
    ------
    ValueError
        "Pass 2, 3, or 2/3 for dimension argument."
    """

    def decorator(func):
        docstring = f"""
            Crosstab {dimension}-dimensional {xarray_obj} to produce Crosstab DataFrame.

            Parameters
            ----------
            agreement_map : {xarray_obj}
                Agreement map, {dimension}-dimensional.

            Returns
            -------
            DataFrame[Crosstab_df]
                Crosstab DataFrame.

            References
            ----------
            .. [1] `flox.xarray.xarray_reduce() <https://flox.readthedocs.io/en/latest/generated/flox.xarray.xarray_reduce.html>`_
            .. [2] `xarray.rio._check_dimensions() <(https://github.com/corteva/rioxarray/blob/9d5975624fa93b76c451457a97b342ba37dfc792/rioxarray/rioxarray.py)>`_
            .. [3] `xr.rio._obj.dims <https://github.com/corteva/rioxarray/blob/9d5975624fa93b76c451457a97b342ba37dfc792/rioxarray/raster_array.py)`>_
            """
        func.__doc__ = docstring.format(dimension, xarray_obj)
        return func

    return decorator


@pa.check_types
@_crosstab_docstring(2, "xr.DataArray")
def _crosstab_2d_DataArrays(
    agreement_map: xr.DataArray,
    band_name: str = "band",
    band_value: Union[str, Number] = 1,
) -> DataFrame[Crosstab_df]:
    """Please see `_crosstab_docstring` function decorator for docstring"""

    is_dsk = False
    if _check_dask_array(agreement_map):
        agreement_map = agreement_map.drop_vars("spatial_ref")
        is_dsk = True

    agreement_map.name = "group"
    ag_dtype = agreement_map.dtype

    if is_dsk:
        agreement_counts = xarray_reduce(
            agreement_map,
            agreement_map,
            engine="numba",
            expected_groups=dask.array.unique(agreement_map.data),
            func="count",
        )
    else:
        agreement_counts = xarray_reduce(
            agreement_map, agreement_map, engine="numba", func="count"
        )

    def not_nan(number):
        return not np.isnan(number)

    # Handle pairing dictionary attribute
    pairing_dict = agreement_map.attrs["pairing_dictionary"]

    rev_dict = {}
    for k, v in pairing_dict.items():
        if np.isnan(v):
            continue
        if v in rev_dict:
            rev_dict[v].append(list(k))
        else:
            rev_dict[v] = [list(k)]

    # reorganize df to follow contingency table schema instead of xarray-spatial conventions
    crosstab_df = pd.DataFrame(
        {
            "candidate_values": [
                [y[0] for y in rev_dict[x]]
                for x in filter(not_nan, agreement_counts.coords["group"].values)
            ],
            "benchmark_values": [
                [y[1] for y in rev_dict[x]]
                for x in filter(not_nan, agreement_counts.coords["group"].values)
            ],
            "agreement_values": list(
                filter(
                    not_nan, agreement_counts.coords["group"].values.astype(ag_dtype)
                )
            ),
            "counts": [
                x
                for x, y in zip(
                    agreement_counts.values.astype(ag_dtype),
                    agreement_counts.coords["group"].values.astype(ag_dtype),
                )
                if not np.isnan(y)
            ],
        }
    )

    # Add all entries that don't exist in crosstab that exist in pairing dictionary with 0 count
    for k, v in agreement_map.attrs["pairing_dictionary"].items():
        if v not in crosstab_df["agreement_values"].values and not np.isnan(v):
            crosstab_df.loc[-1] = [k[0], k[1], v, 0]  # adding a row
            crosstab_df.index = crosstab_df.index + 1

    # Sort and reindex
    crosstab_df.sort_values(["agreement_values"], inplace=True)
    crosstab_df.reset_index()

    def is_iterable(x):
        return x[0] if isinstance(x, Iterable) else x

    # TODO  Resolve the case of multiple candidate/benchmark pairs being mapped to the same agreement
    #  value, for now just take the first pair
    crosstab_df.loc[:, "candidate_values"] = crosstab_df["candidate_values"].apply(
        is_iterable
    )
    crosstab_df.loc[:, "benchmark_values"] = crosstab_df["benchmark_values"].apply(
        is_iterable
    )

    crosstab_df.insert(0, band_name, band_value)

    return crosstab_df


@pa.check_types
@_crosstab_docstring(3, "xr.DataArray")
def _crosstab_3d_DataArrays(agreement_map: xr.DataArray) -> DataFrame[Crosstab_df]:
    """Please see `_crosstab_docstring` function decorator for docstring"""

    """
    NOTE:
    - Use of `xr.rio._obj.dims` or `xr.DataArray.rio._check_dimensions()` to get band name (sometimes called band)
        - is it always called band? may not always be band?
        - are there any other methods for doing this??
        - should this go elsewhere as it might be repeated?
        - Is this necessary? Can we remove? Under what circumstance would it not be "band"
    - xr.DataArray.rio._check_dimensions() only supports 2/3D arrays and throws this error:
        - *** rioxarray.exceptions.TooManyDimensions: Only 2D and 3D data arrays supported.
        - This is ok and is useful to restrict dimensionality.
    """
    # get band name
    band_name_agreement = agreement_map.rio._check_dimensions()

    # get coordinates for bands
    agreement_map_band_coordinates = agreement_map[band_name_agreement].values

    # cycle through extra dim
    previous_crosstab_df = None  # initializing to avoid having unset
    for i, b in enumerate(agreement_map_band_coordinates):
        crosstab_df = _crosstab_2d_DataArrays(
            agreement_map=agreement_map.sel({band_name_agreement: b}),
            band_name=band_name_agreement,
            band_value=b,
        )

        # concats crosstab_dfs across bands
        if i > 0:
            crosstab_df = pd.concat(
                [previous_crosstab_df, crosstab_df], ignore_index=True
            )

        # save df for previous band
        previous_crosstab_df = crosstab_df

    return crosstab_df


@pa.check_types
@_crosstab_docstring("2/3", "xr.DataArray")
def _crosstab_DataArrays(agreement_map: xr.DataArray) -> DataFrame[Crosstab_df]:
    """Please see `_crosstab_docstring` function decorator for docstring"""

    # TODO: these can be predicates and optional exception raising
    # 3d
    if len(agreement_map.shape) == 3:
        crosstab_func = _crosstab_3d_DataArrays
    # 2d
    elif len(agreement_map.shape) == 2:
        crosstab_func = _crosstab_2d_DataArrays
    else:
        raise ValueError(
            "Candidate and benchmark must be both 2 or 3 dimensional only."
        )

    return crosstab_func(agreement_map=agreement_map)


@pa.check_types
@_crosstab_docstring("3", "xr.Dataset")
def _crosstab_Datasets(agreement_map: xr.DataArray) -> DataFrame[Crosstab_df]:
    """Please see `_crosstab_docstring` function decorator for docstring"""

    # gets variable names
    agreement_variable_names = list(agreement_map.data_vars)

    # loop variables
    previous_crosstab_df = None  # initializing to avoid having unset
    for i, b in enumerate(agreement_variable_names):
        # Pass pairing dictionary to variable if necessary
        if (
            agreement_map[b].attrs.get("pairing_dictionary") is None
            and agreement_map.attrs.get("pairing_dictionary") is not None
        ):
            agreement_map[b].attrs["pairing_dictionary"] = agreement_map.attrs[
                "pairing_dictionary"
            ]

        crosstab_df = _crosstab_2d_DataArrays(
            agreement_map=agreement_map[b], band_value=b
        )

        # concats crosstab_dfs across bands
        if i > 0:
            crosstab_df = pd.concat(
                [previous_crosstab_df, crosstab_df], ignore_index=True
            )

        # save df for previous band
        previous_crosstab_df = crosstab_df

    return crosstab_df
