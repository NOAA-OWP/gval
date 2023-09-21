from typing import Union, List, Optional, Tuple

import numpy as np
import dask
import geopandas as gpd
import xarray as xr
from flox.xarray import xarray_reduce
from geocube.api.core import make_geocube


def subsample(
    candidate: Union[xr.DataArray, xr.Dataset],
    benchmark: Union[xr.DataArray, xr.Dataset],
    subsampling_df: Optional[gpd.GeoDataFrame] = None,
) -> Union[
    List[Union[Tuple[xr.DataArray, xr.DataArray], Tuple[xr.Dataset, xr.Dataset]]]
]:
    """
    Method to subsample a candidate and benchmark map

    Parameters
    ----------
    candidate: Union[xr.DataArray, xr.Dataset]
        Candidate map for evaluating
    benchmark: Union[xr.DataArray, xr.Dataset]
        Benchmark map for
    subsampling_df: Optional[gpd.GeoDataFrame], default = None
        Dataframe with columns regarding the geometry and method to subsample

    Raises
    ------
    ValueError
        Sampling_df crs cannot be none

    Returns
    -------
    Union[List[Union[Tuple[xr.DataArray, xr.DataArray], Tuple[xr.Dataset, xr.Dataset]]]]
        Subsampled candidate and benchmark maps or a list of subsampled candidate and benchmark maps

    """

    if subsampling_df.crs is None:
        raise ValueError("sampling_df crs cannot be none")

    sampled_maps = []

    for idx, (_, row) in enumerate(subsampling_df.iterrows()):
        drop, invert = (
            (True, False) if row["subsample_type"] == "include" else (False, True)
        )
        candidate_copy = candidate.rio.clip(
            [row["geometry"]], subsampling_df.crs, drop=drop, invert=invert
        )
        benchmark_copy = benchmark.rio.clip(
            [row["geometry"]], subsampling_df.crs, drop=drop, invert=invert
        )

        if isinstance(candidate, xr.DataArray):
            candidate_copy.attrs["sample_percentage"] = get_subsample_percent(
                subsampling_df.iloc[idx : idx + 1, :], candidate
            )
        else:
            candidate_copy.attrs["sample_percentage"] = [
                get_subsample_percent(
                    subsampling_df.iloc[idx : idx + 1, :],
                    candidate[c_var],
                )
                for c_var in candidate_copy.data_vars
            ]

        # Get sampled maps and percents
        sampled_maps.append([candidate_copy, benchmark_copy])

    return sampled_maps


def get_subsample_percent(
    sampling_df: gpd.GeoDataFrame, og_data: Union[xr.DataArray, xr.Dataset]
) -> float:
    """
    Get percent of original map subsampled by geometry

    Parameters
    ----------
    sampling_df: gpd.GeoDataFrame
        Dataframe with subsample geometries and options
    og_data: Union[xr.DataArray, xr.Dataset]
        Original nodata count

    Returns
    -------
    float
        Percentage of original data the sample covers

    """
    total_size = (
        dask.array.prod(og_data.shape[-2:])
        if og_data.chunks is not None
        else np.prod(og_data.shape[-2:])
    )

    mask = make_geocube(sampling_df, measurements=["subsample_id"], like=og_data)

    try:
        percent = float(
            xarray_reduce(mask["subsample_id"], mask["subsample_id"], func="count")
            / total_size
            * 100
        )
    except IndexError as e:  # pragma: no cover
        raise e("No spatial overlap in subsample geometry and candidate/benchmark data")

    del mask

    return percent
