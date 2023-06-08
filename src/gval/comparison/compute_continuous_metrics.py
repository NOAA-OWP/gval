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

from gval.utils.schemas import Metrics_df

from gval.statistics.continuous_stat_funcs import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_percentage_error,
    mean_absolute_percentage_error,
    coefficient_of_determination,
    mean_normalized_mean_absolute_error,
    range_normalized_mean_absolute_error,
    mean_normalized_root_mean_squared_error,
    range_normalized_root_mean_squared_error,
)


@pa.check_types
def _compute_continuous_metrics(
    agreement_map: Union[xr.DataArray, xr.Dataset] = None,
    candidate_map: Union[xr.DataArray, xr.Dataset] = None,
    benchmark_map: Union[xr.DataArray, xr.Dataset] = None,
    metrics: Union[str, Iterable[str]] = "all",
) -> DataFrame[Metrics_df]:
    """
    Computes categorical metrics from a crosstab df.

    Parameters
    ----------
    agreement_map : Union[xr.DataArray, xr.Dataset], default = None
        Agreement map, error based (candidate - benchmark).
    candidate_map : Union[xr.DataArray, xr.Dataset], default = None
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset], default = None
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

    all_metric_funcs = [
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error,
        mean_percentage_error,
        mean_absolute_percentage_error,
        coefficient_of_determination,
        mean_normalized_mean_absolute_error,
        range_normalized_mean_absolute_error,
        mean_normalized_root_mean_squared_error,
        range_normalized_root_mean_squared_error,
    ]

    all_metric_names = [
        "mean_absolute_error",
        "mean_squared_error",
        "root_mean_squared_error",
        "mean_percentage_error",
        "mean_absolute_percentage_error",
        "coefficient_of_determination",
        "normalized_mean_absolute_error",
        "normalized_root_mean_squared_error",
    ]

    # create dictionary of metric names and functions
    all_metric_dict = dict(zip(all_metric_names, all_metric_funcs))

    if metrics == "all":
        metric_dict = all_metric_dict
    else:
        metrics_set = set(metrics)
        metric_dict = {k: all_metric_dict[k] for k in metrics if k in metrics_set}

    # create metrics_df
    metric_df = dict()
    for name, func in metric_dict.items():
        metric_df[name] = func(agreement_map, candidate_map, benchmark_map)

    def is_nested_dict(d):
        if not isinstance(d, dict):
            return False
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
