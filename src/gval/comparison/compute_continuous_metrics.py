"""
Computes continuous value metrics given an agreement map.
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Iterable, Union

import pandera as pa
import pandas as pd
from pandera.typing import DataFrame
import xarray as xr

from gval import ContStats
from gval.utils.schemas import Metrics_df


@pa.check_types
def _compute_continuous_metrics(
    agreement_map: Union[xr.DataArray, xr.Dataset],
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    metrics: Union[str, Iterable[str]] = "all",
) -> DataFrame[Metrics_df]:
    """
    Computes categorical metrics from a crosstab df.

    Parameters
    ----------
    agreement_map : Union[xr.DataArray, xr.Dataset]
        Agreement map, error based (candidate - benchmark).
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.
    metrics : Union[str, Iterable[str]], default = "all"
        String or list of strings representing metrics to compute.

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
    .. [1] [7th International Verification Methods Workshop](https://www.cawcr.gov.au/projects/verification/)
    .. [2] [3.3. Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
    """

    # compute error based metrics such as MAE, MSE, RMSE, etc. from agreement map and produce metrics_df
    statistics, names = ContStats.process_statistics(
        metrics,
        error=agreement_map,
        candidate_map=candidate_map,
        benchmark_map=benchmark_map,
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

    return metric_df
