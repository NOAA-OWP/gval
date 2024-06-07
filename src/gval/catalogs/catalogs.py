"""
Compare catalogs of candidates and benchmarks.
"""
from __future__ import annotations

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import gc
from typing import Iterable, Optional, Callable, Tuple
import os

import pandas as pd
from rioxarray import open_rasterio as rxr_or
import xarray as xr
import dask.dataframe as dd


def catalog_compare(
    candidate_catalog: pd.DataFrame | dd.DataFrame,
    benchmark_catalog: pd.DataFrame | dd.DataFrame,
    map_ids: str | Iterable[str],
    how: str = "inner",
    on: Optional[str | Iterable[str]] = None,
    left_on: Optional[str | Iterable[str]] = None,
    right_on: Optional[str | Iterable[str]] = None,
    suffixes: tuple[str, str] = ("_candidate", "_benchmark"),
    merge_kwargs: Optional[dict] = None,
    open_kwargs: Optional[dict] = None,
    compare_type: str | Callable = "continuous",
    compare_kwargs: Optional[dict] = None,
    agreement_map_field: Optional[str] = None,
    agreement_map_write_kwargs: Optional[dict] = None,
) -> pd.DataFrame | dd.DataFrame:
    """
    Compare catalogs of candidate and benchmark maps.

    Parameters
    ----------
    candidate_catalog : pandas.DataFrame | dask.DataFrame
        Candidate catalog.
    benchmark_catalog : pandas.DataFrame | dask.DataFrame
        Benchmark catalog.
    map_ids : str | Iterable of str
        Column name(s) where maps or paths to maps occur. If str is given, then the same value should occur in both catalogs. If Iterable[str] is given of length 2, then the column names where maps are will be in [candidate, benchmark] respectively.

        The columns corresponding to map_ids should have either str, xarray.DataArray, xarray.Dataset, rasterio.io.DatasetReader, rasterio.vrt.WarpedVRT, or os.PathLike objects.
    how : str, default = "inner"
        Type of merge to perform. See pandas.DataFrame.merge for more information.
    on : str | Iterable of str, default = None
        Column(s) to join on. Must be found in both catalogs. If None, and left_on and right_on are also None, then the intersection of the columns in both catalogs will be used.
    left_on : str | Iterable of str, default = None
        Column(s) to join on in left catalog. Must be found in left catalog.
    right_on : str | Iterable of str, default = None
        Column(s) to join on in right catalog. Must be found in right catalog.
    suffixes : tuple of str, default = ("_candidate", "_benchmark")
        Suffixes to apply to overlapping column names in candidate and benchmark catalogs, respectively. Length two tuple of strings.
    merge_kwargs : dict, default = None
        Keyword arguments to pass to pandas.DataFrame.merge.
    compare_type : str | Callable, default = "continuous"
        Type of comparison to perform. If str, then must be one of {"continuous", "categorical", "probabilistic"}. If Callable, then must be a function that takes two xarray.DataArray or xarray.Dataset objects and returns a tuple of length 2. The first element of the tuple must be an xarray.DataArray or xarray.Dataset object representing the agreement map. The second element of the tuple must be a pandas.DataFrame object representing the metrics.
    compare_kwargs : dict, default = None
        Keyword arguments to pass to the compare_type function.
    agreement_map_field : str, default = None
        Column name to write agreement maps to. If None, then agreement maps will not be written to file.
    agreement_map_write_kwargs : dict, default = None
        Keyword arguments to pass to xarray.DataArray.rio.to_raster when writing agreement maps to file.

    Raises
    ------
    ValueError
        If map_ids is not str or Iterable of str.
        If compare_type is not str or Callable.
        If compare_type is str and not one of {"continuous", "categorical", "probabilistic"}.
    NotImplementedError
        If compare_type is "probabilistic".

    Returns
    -------
    pandas.DataFrame | dask.DataFrame
        Agreement catalog.
    """

    # unpack map_ids
    if isinstance(map_ids, str):
        candidate_map_ids, benchmark_map_ids = map_ids, map_ids
    elif isinstance(map_ids, Iterable):
        candidate_map_ids, benchmark_map_ids = map_ids
    else:
        raise ValueError("map_ids must be str or Iterable of str")

    # set merge_kwargs to empty dict if None
    if merge_kwargs is None:
        merge_kwargs = dict()

    # create agreement catalog
    agreement_catalog = candidate_catalog.merge(
        benchmark_catalog,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
        **merge_kwargs,
    )

    def compare_row(
        row,
        compare_type: str | Callable,
        compare_kwargs: dict,
        open_kwargs: dict,
        agreement_map_field: str,
        agreement_map_write_kwargs: dict,
    ) -> Tuple[xr.DataArray | xr.Dataset, pd.DataFrame]:
        """Compares catalog and benchmark maps by rows"""

        def loadxr(map, open_kwargs):
            """load xarray object if not already"""
            return (
                map
                if isinstance(map, (xr.DataArray, xr.Dataset))
                else rxr_or(map, **open_kwargs)
            )

        # load maps
        candidate_map = loadxr(row[candidate_map_ids + suffixes[0]], open_kwargs)
        benchmark_map = loadxr(row[benchmark_map_ids + suffixes[1]], open_kwargs)

        # set compare_kwargs to empty dict if None
        if compare_kwargs is None:
            compare_kwargs = dict()

        # set agreement_map_write_kwargs to empty dict if None
        if agreement_map_write_kwargs is None:
            agreement_map_write_kwargs = dict()

        if isinstance(compare_type, str):
            if compare_type == "categorical":
                results = candidate_map.gval.categorical_compare(
                    benchmark_map, **compare_kwargs
                )

                # results is a tuple of length 3 or 4
                # agreement_map, crosstab_df, metrics_df, attrs_df = results
                # where attrs_df is optional
                agreement_map, metrics_df = results[0], results[2]

            elif compare_type == "continuous":
                results = candidate_map.gval.continuous_compare(
                    benchmark_map, **compare_kwargs
                )

                # results is a tuple of length 2 or 3
                # agreement_map, metrics_df, attrs_df = results
                # where attrs_df is optional
                agreement_map, metrics_df = results[:2]

            elif compare_type == "probabilistic":
                raise NotImplementedError(
                    "probabilistic comparison not implemented yet"
                )

            else:
                raise ValueError(
                    "compare_type of type str must be one of {'continuous', 'categorical', 'probabilistic'}"
                )

        elif isinstance(compare_type, Callable):
            agreement_map, metrics_df = compare_type(
                candidate_map, benchmark_map, **compare_kwargs
            )

        else:
            raise ValueError("compare_type must be str or Callable")

        # Write agreement map to file
        if (agreement_map_field is not None) & isinstance(
            agreement_map, (xr.DataArray, xr.Dataset)
        ):
            if isinstance(row[agreement_map_field], (str, os.PathLike)):
                agreement_map.rio.to_raster(
                    row[agreement_map_field], **agreement_map_write_kwargs
                )

        # Unfortunately necessary until a fix is found in xarray/rioxarray io
        del candidate_map, benchmark_map, agreement_map
        gc.collect()

        return metrics_df

    # make kwargs for dask apply
    if isinstance(agreement_catalog, dd.DataFrame):
        dask_kwargs = {"meta": ("output", "f8")}
    else:
        dask_kwargs = {}

    # set open_kwargs to empty dict if None
    if open_kwargs is None:
        open_kwargs = dict()

    # apply compare_row to each row of agreement_catalog
    metrics_df = agreement_catalog.apply(
        compare_row,
        axis=1,
        **dask_kwargs,
        compare_type=compare_type,
        open_kwargs=open_kwargs,
        compare_kwargs=compare_kwargs,
        agreement_map_field=agreement_map_field,
        agreement_map_write_kwargs=agreement_map_write_kwargs,
    )

    def nested_merge(i, sub_df) -> pd.DataFrame:
        """Duplicated agreement row for each row in sub_df"""
        try:
            agreement_row = agreement_catalog.iloc[i].to_frame().T
        except NotImplementedError:
            agreement_row = agreement_catalog.loc[agreement_catalog.index == i]

        sub_df.index = [i] * len(sub_df)
        return agreement_row.join(sub_df)

    # merge agreement_catalog with metrics_df
    if isinstance(metrics_df, dd.Series):
        return dd.concat(
            [nested_merge(i, sub_df) for i, sub_df in enumerate(metrics_df)]
        ).reset_index(drop=True)

    if isinstance(metrics_df, pd.Series):
        return pd.concat(
            [nested_merge(i, sub_df) for i, sub_df in enumerate(metrics_df)]
        ).reset_index(drop=True)
