"""
Computes continuous value metrics given an agreement map.
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Iterable, Union, List

import numpy as np
import pandera.pandas as pa
import pandas as pd
from pandera.typing import DataFrame
import xarray as xr
import geopandas as gpd
import dask as da

from gval import ContStats
from gval.utils.schemas import Metrics_df, Subsample_identifiers, Sample_identifiers
from gval.utils.loading_datasets import _convert_to_dataset


def _get_masked_data(
    agreement: xr.Dataset,
    candidate: xr.Dataset,
    benchmark: xr.Dataset,
    nodata: list,
    var_name: str,
) -> List[xr.Dataset]:
    """
    Gets masked data for integer valued datasets in order to not process nodata values

    Parameters
    ----------
    agreement : xr.Dataset
        Agreement Map
    candidate : xr.Dataset
        Candidate Map
    benchmark : xr.Dataset
        Benchmark Map
    nodata : list
        Nodata values in the list
    var_name : str
        Name of variable

    Returns
    -------
    List[xr.Dataset, xr.Dataset, xr.Dataset]
        Datasets with selected coordinates
    """

    cmask, bmask = (
        xr.where(candidate[var_name] == nodata, 0, 1),
        xr.where(benchmark[var_name] == nodata, 0, 1),
    )
    tmask = cmask & bmask

    # Select coordinates from xarray
    with da.config.set({"array.slicing.split_large_chunks": True}):
        agreement_sel = agreement[var_name].where(tmask.compute(), drop=True)
        candidate_sel = candidate[var_name].where(tmask.compute(), drop=True)
        benchmark_sel = benchmark[var_name].where(tmask.compute(), drop=True)

    return (
        agreement_sel,
        candidate_sel,
        benchmark_sel,
    )


@pa.check_types
def _compute_continuous_metrics(
    agreement_map: Union[xr.DataArray, xr.Dataset],
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    metrics: Union[str, Iterable[str]] = "all",
    subsampling_average: str = "micro",
    subsampling_df: gpd.GeoDataFrame = None,
) -> DataFrame[Metrics_df]:
    """
    Computes continuous metrics.

    Parameters
    ----------
    agreement_map : Union[xr.DataArray, xr.Dataset, List[Union[xr.DataArray, xr.Dataset]]]
        Agreement map, error based (candidate - benchmark).
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.
    metrics : Union[str, Iterable[str]], default = "all"
        String or list of strings representing metrics to compute.
    subsampling_average : str, default = "micro"
        Strategy to average samples if there is more than one in the agreement map
    subsampling_df : gpd.GeoDataFrame, default = None
        DataFrame with geometries to subsample or use as exclusionary masks and optional sample weights

    Returns
    -------
    DataFrame[Metrics_df]
        Metrics DF with computed metrics per sample.

    Raises
    ------
    ValueError
        If metrics is not a string or list of strings.

    References
    ----------
    .. [1] `7th International Verification Methods Workshop <https://www.cawcr.gov.au/projects/verification/>`_
    .. [2] `3.3. Metrics and scoring: quantifying the quality of predictions <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
    """

    if not isinstance(agreement_map, list):
        agreement_map = [agreement_map]
        candidate_map = [candidate_map]
        benchmark_map = [benchmark_map]

    metric_dfs = []
    for idx, (agreement, benchmark, candidate) in enumerate(
        zip(agreement_map, benchmark_map, candidate_map)
    ):
        # Change data to Dataset if DataArray
        agreement = _convert_to_dataset(agreement)
        candidate = _convert_to_dataset(candidate)
        benchmark = _convert_to_dataset(benchmark)

        # Check if integer type and nodata values
        is_int = (
            np.issubdtype(candidate.dtype, np.integer)
            if isinstance(candidate, xr.DataArray)
            else np.issubdtype(candidate["band_1"].dtype, np.integer)
        )
        nodata = [agreement[x].rio.nodata for x in agreement.data_vars]

        # Remove no data value if int type form calculation, otherwise leave all values in
        # Necessary because there is not an int sentinel value
        if is_int and np.all([x is not None for x in nodata]):
            final_stats = []
            # Iterate through each band and gather statistics
            for nodata_idx, var_name in enumerate(agreement.data_vars):
                # Create mask for all nodata values
                agreement_sel, candidate_sel, benchmark_sel = _get_masked_data(
                    agreement, candidate, benchmark, nodata[nodata_idx], var_name
                )

                statistics, names = ContStats.process_statistics(
                    metrics,
                    error=agreement_sel,
                    candidate_map=candidate_sel,
                    benchmark_map=benchmark_sel,
                )

                del agreement_sel, candidate_sel, benchmark_sel

                final_stats.append(statistics)

            statistics = [
                {f"band_{idx + 1}": val for idx, val in enumerate(lst)}
                for lst in np.array(final_stats).T
            ]

        else:
            statistics, names = ContStats.process_statistics(
                metrics,
                error=agreement,
                candidate_map=candidate,
                benchmark_map=benchmark,
            )

        # create metrics_df
        metric_df = dict()
        for name, stat in zip(names, statistics):
            metric_df[name] = stat

        def is_nested_dict(d):
            # if not isinstance(d, dict):
            #     return False
            return any(isinstance(v, dict) for v in d.values())

        if is_nested_dict(metric_df):
            metric_df = pd.DataFrame.from_dict(metric_df, orient="index").transpose()
            metric_df.reset_index(inplace=True)
            metric_df.rename(columns={"index": "band"}, inplace=True)
            metric_df["band"] = metric_df["band"].str.replace("band_", "")

        else:
            # dataarray
            metric_df = pd.DataFrame(metric_df, index=[0])

            # add band
            metric_df.insert(0, "band", "1")

        if subsampling_df is not None:
            metric_df.insert(0, "subsample", f"{idx + 1}")

        metric_dfs.append(metric_df)

    metric_df = pd.concat(metric_dfs).reset_index().drop(columns=["index"])

    if subsampling_df is not None:
        if subsampling_average == "band":
            metric_df = (
                metric_df.groupby(Subsample_identifiers.columns())
                .mean(numeric_only=True)
                .reset_index()
            )

            metric_df.insert(1, "band", "averaged")

        if subsampling_average == "subsample":
            metric_df = (
                metric_df.groupby(Sample_identifiers.columns())
                .mean(numeric_only=True)
                .reset_index()
            )

            metric_df.insert(0, "subsample", "averaged")

        if subsampling_average == "weighted":
            if subsampling_df.get("weights") is None:
                raise ValueError(
                    "Must have weights if weighted is chosen for subsampling"
                )  # pragma: no cover

            metric_df.loc[:, "weights"] = subsampling_df["weights"]

            # compute weighted average
            weighted_metrics = (
                metric_df.loc[:, metrics]
                .multiply(metric_df.loc[:, "weights"], axis=0)
                .reset_index(drop=True)
            )

            # add weighted metrics to metric_df
            metric_df.loc[:, metrics] = weighted_metrics

            # take average of weighted metrics
            metric_df = (
                metric_df.groupby(Sample_identifiers.columns())
                .sum(numeric_only=True)
                .drop(
                    columns=["weights", "subsample"],
                    errors="ignore",
                )
                .divide(metric_df.loc[:, "weights"].sum())
                .reset_index()
            )

            metric_df.insert(0, "subsample", "averaged")

    return metric_df
