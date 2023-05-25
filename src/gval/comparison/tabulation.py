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

from typing import Iterable, Optional, Union, Callable
from numbers import Number

import numpy as np
import pandas as pd
from xrspatial.zonal import crosstab
import xarray as xr
import pandera as pa
from pandera.typing import DataFrame
import dask

from gval.utils.schemas import Xrspatial_crosstab_df, Crosstab_df
from gval.homogenize.spatial_alignment import _check_dask_array


@pa.check_types
def _convert_crosstab_to_contigency_table(
    crosstab_df: DataFrame[Xrspatial_crosstab_df],
    band_name: str,
    band_value: Union[str, Number],
) -> DataFrame[Crosstab_df]:
    """
    Reorganizes crosstab output to Crosstab 2D DataFrame format.

    ----------
    Parameters
    crosstab_df : DataFrame[Xrspatial_crosstab_df]
        Output DataFrame from :func:`xarray-spatial.zonal.crosstab`.

    Returns
    -------
    DataFrame[Crosstab_df]
        Crosstab DataFrame using candidate and benchmark conventions.
    """
    if isinstance(crosstab_df, dask.dataframe.core.DataFrame):
        crosstab_df = crosstab_df.compute()

    # renames zone, renames column index, melts dataframe, then resets the index.
    crosstab_df = (
        crosstab_df.rename(columns={"zone": "candidate_values"})
        .rename_axis(columns="benchmark_values")
        .melt(id_vars="candidate_values", value_name="counts", ignore_index=False)
        .reset_index(drop=True)
    )

    # add band column
    crosstab_df.insert(0, band_name, band_value)

    return crosstab_df


@pa.check_types
def _compute_agreement_values(
    crosstab_df: DataFrame[Crosstab_df],
    comparison_function: Callable[..., float],
) -> DataFrame[Crosstab_df]:
    """
    Computes agreement values from Crosstab DataFrame.

    Parameters
    ----------
    crosstab_df : DataFrame[Crosstab_df]
        Crosstab DataFrame.
    comparison_function : Callable[[float, float], float]
        Function to compute agreement values.

    Returns
    -------
    DataFrame[Crosstab_df]
        Crosstab DataFrame with agreement values.
    """

    def apply_pairing_function(row):
        return comparison_function(row["candidate_values"], row["benchmark_values"])

    # copy crosstab_df
    crosstab_df = crosstab_df.copy()

    agreement_values = crosstab_df.apply(apply_pairing_function, axis=1)

    crosstab_df.insert(3, "agreement_values", agreement_values)

    return crosstab_df


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
            candidate_map : {xarray_obj}
                Candidate map, {dimension}-dimensional.
            benchmark_map : {xarray_obj}
                Benchmark map, {dimension}-dimensional.
            allow_candidate_values : Optional[Iterable[Union[int,float]]], default = None
                Sequence of values in candidate to include in crosstab. Remaining values are excluded.
            allow_benchmark_values : Optional[Iterable[Union[int,float]]], default = None
                Sequence of values in benchmark to include in crosstab. Remaining values are excluded.
            exclude_value : Optional[Number], default = None
                Value to exclude from crosstab. This could be used to denote a no data value if masking wasn't used. By default, NaNs are not cross-tabulated.
            comparison_function : Callable[[float, float], float], default = None
                Function to compute agreement values. If None, then no agreement values are computed.

            Returns
            -------
            DataFrame[Crosstab_df]
                Crosstab DataFrame.

            References
            ----------
            .. [1] :func:[`xrspatial.zonal.crosstab`](https://xarray-spatial.org/reference/_autosummary/xrspatial.zonal.crosstab.html)
            .. [2] [xarray.rio._check_dimensions()](https://github.com/corteva/rioxarray/blob/9d5975624fa93b76c451457a97b342ba37dfc792/rioxarray/rioxarray.py)
            .. [3] [xr.rio._obj.dims](https://github.com/corteva/rioxarray/blob/9d5975624fa93b76c451457a97b342ba37dfc792/rioxarray/raster_array.py)
            """
        func.__doc__ = docstring.format(dimension, xarray_obj)
        return func

    return decorator


@pa.check_types
@_crosstab_docstring(2, "xr.DataArray")
def _crosstab_2d_DataArrays(
    candidate_map: xr.DataArray,
    benchmark_map: xr.DataArray,
    band_name: str = "band",
    band_value: Union[str, Number] = 1,
    allow_candidate_values: Optional[Iterable[Number]] = None,
    allow_benchmark_values: Optional[Iterable[Number]] = None,
    exclude_value: Optional[Number] = None,
    comparison_function: Optional[Callable[..., float]] = None,
) -> DataFrame[Crosstab_df]:
    """Please see `_crosstab_docstring` function decorator for docstring"""

    if _check_dask_array(candidate_map):
        candidate_map = candidate_map.drop("spatial_ref")
        benchmark_map = benchmark_map.drop("spatial_ref")

    crosstab_df = crosstab(
        zones=candidate_map,
        values=benchmark_map,
        zone_ids=allow_candidate_values,
        cat_ids=allow_benchmark_values,
        nodata_values=exclude_value,
    )

    # reorganize df to follow contingency table schema instead of xarray-spatial conventions
    crosstab_df = _convert_crosstab_to_contigency_table(
        crosstab_df, band_name, band_value
    )

    # insert agreement values
    if comparison_function is not None:
        crosstab_df = _compute_agreement_values(crosstab_df, comparison_function)

    return crosstab_df


@pa.check_types
@_crosstab_docstring(3, "xr.DataArray")
def _crosstab_3d_DataArrays(
    candidate_map: xr.DataArray,
    benchmark_map: xr.DataArray,
    allow_candidate_values: Optional[Iterable[Number]] = None,
    allow_benchmark_values: Optional[Iterable[Number]] = None,
    exclude_value: Optional[Number] = None,
    comparison_function: Optional[Callable[..., float]] = None,
) -> DataFrame[Crosstab_df]:
    """Please see `_crosstab_docstring` function decorator for docstring"""

    # check number of dimensions
    assert (
        len(candidate_map.shape) == len(benchmark_map.shape) == 3
    ), "Candidate and benchmark must both be 3-dimensional"

    # check dimensionality
    # Is this necessary or is this done with functionality that checks band coordinates and within crosstab()
    # assert (
    #    candidate_map.shape == benchmark_map.shape
    # ), f"Dimensionalities of candidate {candidate_map.shape} and benchmark {benchmark_map.#shape} must match."

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
    band_name_candidate = candidate_map.rio._check_dimensions()
    band_name_benchmark = benchmark_map.rio._check_dimensions()

    # get coordinates for bands
    candidate_map_band_coordinates = candidate_map[band_name_candidate].values
    benchmark_map_band_coordinates = benchmark_map[band_name_benchmark].values

    # check coordinates of extra dim to make sure they are the same
    # TODO: Having dataarrays with different extra dim coords remains untested.
    np.testing.assert_equal(
        candidate_map_band_coordinates,
        benchmark_map_band_coordinates,
        f"Coordinates of candidate dimension, {band_name_candidate}, and benchmark, {band_name_benchmark}, must be the same.",
    )

    # cycle through extra dim
    previous_crosstab_df = None  # initializing to avoid having unset
    for i, b in enumerate(candidate_map_band_coordinates):
        crosstab_df = _crosstab_2d_DataArrays(
            candidate_map=candidate_map.sel({band_name_candidate: b}),
            benchmark_map=benchmark_map.sel({band_name_benchmark: b}),
            band_name=band_name_candidate,
            band_value=b,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
            exclude_value=exclude_value,
            comparison_function=comparison_function,
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
def _crosstab_DataArrays(
    candidate_map: xr.DataArray,
    benchmark_map: xr.DataArray,
    allow_candidate_values: Optional[Iterable[Number]] = None,
    allow_benchmark_values: Optional[Iterable[Number]] = None,
    exclude_value: Optional[Number] = None,
    comparison_function: Optional[Callable[..., float]] = None,
) -> DataFrame[Crosstab_df]:
    """Please see `_crosstab_docstring` function decorator for docstring"""

    # TODO: these can be predicates and optional exception raising
    # 3d
    if len(candidate_map.shape) == len(benchmark_map.shape) == 3:
        crosstab_func = _crosstab_3d_DataArrays
    # 2d
    elif len(candidate_map.shape) == len(benchmark_map.shape) == 2:
        crosstab_func = _crosstab_2d_DataArrays
    else:
        raise ValueError(
            "Candidate and benchmark must be both 2 or 3 dimensional only."
        )

    return crosstab_func(
        candidate_map=candidate_map,
        benchmark_map=benchmark_map,
        allow_candidate_values=allow_candidate_values,
        allow_benchmark_values=allow_benchmark_values,
        exclude_value=exclude_value,
        comparison_function=comparison_function,
    )


@pa.check_types
@_crosstab_docstring("3", "xr.Dataset")
def _crosstab_Datasets(
    candidate_map: xr.Dataset,
    benchmark_map: xr.Dataset,
    allow_candidate_values: Optional[Iterable[Number]] = None,
    allow_benchmark_values: Optional[Iterable[Number]] = None,
    exclude_value: Optional[Number] = None,
    comparison_function: Optional[Callable[..., float]] = None,
) -> DataFrame[Crosstab_df]:
    """Please see `_crosstab_docstring` function decorator for docstring"""

    if _check_dask_array(candidate_map):
        # TODO:  Currently there is an issue open on xarray spatial regarding dask dataset useage in crosstab
        # https://github.com/makepath/xarray-spatial/issues/777
        candidate_map = candidate_map.compute()
        benchmark_map = benchmark_map.compute()

    # gets variable names
    candidate_variable_names = list(candidate_map.data_vars)
    benchmark_variable_names = list(benchmark_map.data_vars)

    # checks matching variable names
    # TODO: Is this desired? Should we just check for matching variable lengths?
    np.testing.assert_equal(
        candidate_variable_names,
        benchmark_variable_names,
        "Variable names must match for candidate and benchmark",
    )

    # loop variables
    previous_crosstab_df = None  # initializing to avoid having unset
    for i, b in enumerate(candidate_variable_names):
        crosstab_df = _crosstab_2d_DataArrays(
            candidate_map=candidate_map[b],
            benchmark_map=benchmark_map[b],
            band_value=b,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
            exclude_value=exclude_value,
            comparison_function=comparison_function,
        )

        # concats crosstab_dfs across bands
        if i > 0:
            crosstab_df = pd.concat(
                [previous_crosstab_df, crosstab_df], ignore_index=True
            )

        # save df for previous band
        previous_crosstab_df = crosstab_df

    return crosstab_df
