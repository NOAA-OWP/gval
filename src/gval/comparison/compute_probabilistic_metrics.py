"""
Computes probabilistic metrics.
"""

__author__ = "Fernando Aristizabal"

from typing import Optional, Any, Union

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame
import xskillscore as xs
import xarray as xr
import warnings

from gval.utils.schemas import Prob_metrics_df

PROB_METRICS = {
    "brier_score": xs.brier_score,
    "crps_ensemble": xs.crps_ensemble,
    "crps_gaussian": xs.crps_gaussian,
    "crps_quadrature": xs.crps_quadrature,
    "discrimination": xs.discrimination,
    "rank_histogram": xs.rank_histogram,
    "reliability": xs.reliability,
    "roc": xs.roc,
    "rps": xs.rps,
    "threshold_brier_score": xs.threshold_brier_score,
}

# None values are either used for required arguments or for optional arguments that are None by default
DEFAULT_METRIC_KWARGS = {
    "brier_score": {
        "observations": None,
        "forecasts": None,
        "member_dim": "member",
        "fair": False,
        "dim": None,
        "weights": None,
        "keep_attrs": False,
    },
    "crps_ensemble": {
        "observations": None,
        "forecasts": None,
        "member_weights": None,
        "issorted": False,
        "member_dim": "member",
        "dim": None,
        "weights": None,
        "keep_attrs": False,
    },
    "crps_gaussian": {
        "observations": None,
        "mu": None,
        "sig": None,
        "dim": None,
        "weights": None,
        "keep_attrs": False,
    },
    "crps_quadrature": {
        "observations": None,
        "cdf_or_dist": None,
        "xmin": None,
        "xmax": None,
        "tol": 1e-6,
        "dim": None,
        "keep_attrs": False,
    },
    "discrimination": {
        "observations": None,
        "forecasts": None,
        "dim": None,
        "probability_bin_edges": np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    },
    "rank_histogram": {
        "observations": None,
        "forecasts": None,
        "dim": None,
        "member_dim": "member",
    },
    "reliability": {
        "observations": None,
        "forecasts": None,
        "dim": None,
        "probability_bin_edges": np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        "keep_attrs": False,
    },
    "roc": {
        "observations": None,
        "forecasts": None,
        "bin_edges": "continuous",
        "dim": None,
        "drop_intermediate": False,
        "return_results": "area",
    },
    "rps": {
        "observations": None,
        "forecasts": None,
        "category_edges": None,
        "dim": None,
        "fair": False,
        "weights": None,
        "keep_attrs": False,
        "member_dim": "member",
        "input_distributions": None,
    },
    "threshold_brier_score": {
        "observations": None,
        "forecasts": None,
        "threshold": None,
        "issorted": False,
        "member_dim": "member",
        "dim": None,
        "weights": None,
        "keep_attrs": False,
    },
}

REQUIRED_METRIC_KWARGS = {
    "brier_score": {"observations", "forecasts"},
    "crps_ensemble": {"observations", "forecasts"},
    "crps_gaussian": {"observations", "mu", "sig"},
    "crps_quadrature": {"observations", "cdf_or_dist"},
    "discrimination": {"observations", "forecasts"},
    "rank_histogram": {"observations", "forecasts"},
    "reliability": {"observations", "forecasts"},
    "roc": {"observations", "forecasts"},
    "rps": {"observations", "forecasts", "category_edges"},
    "threshold_brier_score": {"observations", "forecasts", "threshold"},
}


@pa.check_types
def _compute_probabilistic_metrics(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    metric_kwargs: dict,
    return_on_error: Optional[Any] = None,
) -> DataFrame[Prob_metrics_df]:
    """
    Computes probabilistic metrics from candidate and benchmark maps.

    Parameters
    ----------
    candidate_map : xr.DataArray or xr.Dataset
        Candidate map.
    benchmark_map : xr.DataArray or xr.Dataset
        Benchmark map.
    metric_kwargs : dict
        Dictionary of keyword arguments to metric functions. Keys must be metrics. Values are keyword arguments to metric functions. Don't pass keys or values for 'observations' or 'forecasts' as these are handled internally with `benchmark_map` and `candidate_map`, respectively. Available keyword arguments by metric are available in `DEFAULT_METRIC_KWARGS`. If values are None or empty dictionary, default values in `DEFAULT_METRIC_KWARGS` are used.
    return_on_error : Optional[Any], default = None
        Value to return within metrics dataframe if an error occurs when computing a metric. If None, the metric is not computed and None is returned. If 'error', the raised error is returned.

    Returns
    -------
    DataFrame[Prob_metrics_df]
        Probabilistic metrics Pandas DataFrame with computed xarray's per metric and sample.

    Raises
    ------
    ValueError
        If keyword argument is required for metric but not passed.
        If keyword argument is not available for metric but passed.
        If metric is not available.

    Warns
    -----
    UserWarning
        Warns if a metric cannot be computed. `return_on_error` determines whether the metric is not computed and None is returned or if the raised error is returned.

    References
    ----------
    .. [1] `Scoring rule <https://en.wikipedia.org/wiki/Scoring_rule#Examples_of_proper_scoring_rules>`_
    .. [2] `Strictly Proper Scoring Rules, Prediction, and Estimation <https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf>`_
    .. [3] `Properscoring/properscoring <https://github.com/properscoring/properscoring>`_
    .. [4] `xskillscore/xskillscore <https://xskillscore.readthedocs.io/en/stable/api.html#probabilistic-metrics>`_
    .. [5] `What is a proper scoring rule? <https://statisticaloddsandends.wordpress.com/2021/03/27/what-is-a-proper-scoring-rule/>`_
    """

    # Input handling for metric_kwargs variable ##########################################

    # ensure metric_kwargs are in REQUIRED_METRIC_KWARGS and DEFAULT_METRIC_KWARGS
    # also handle cases where values in metric_kwargs are None
    for metric in metric_kwargs.keys():
        # metric is available
        if metric in DEFAULT_METRIC_KWARGS.keys():
            # handle cases where values in metric_kwargs are None or empty
            if (metric_kwargs[metric] is None) | (metric_kwargs[metric] == dict()):
                for kw, val in DEFAULT_METRIC_KWARGS[metric].items():
                    kw_required = kw in REQUIRED_METRIC_KWARGS[metric]
                    dont_ignore_kwarg = kw not in {"observations", "forecasts"}

                    # when kw is required and not to be ignored
                    if (kw_required) & (dont_ignore_kwarg):
                        raise ValueError(
                            f"Keyword argument '{kw}' required for metric {metric}."
                        )

                    # if kw is optional and not to be ignored
                    if (not kw_required) & (dont_ignore_kwarg):
                        if metric_kwargs[metric] is None:
                            metric_kwargs[metric] = dict()

                        metric_kwargs[metric][kw] = val

            # add in observations if needed
            if "observations" in DEFAULT_METRIC_KWARGS[metric]:
                metric_kwargs[metric]["observations"] = benchmark_map

            # add in forecasts if needed
            if "forecasts" in DEFAULT_METRIC_KWARGS[metric]:
                metric_kwargs[metric]["forecasts"] = candidate_map

            # make sure all required arguments are passed
            for kw in REQUIRED_METRIC_KWARGS[metric]:
                if kw not in metric_kwargs[metric].keys():
                    raise ValueError(
                        f"Keyword argument '{kw}' required for metric {metric}."
                    )

            # make sure no extra arguments are passed
            for kw in metric_kwargs[metric].keys():
                if kw not in DEFAULT_METRIC_KWARGS[metric].keys():
                    raise ValueError(
                        f"Keyword argument '{kw}' not available for metric {metric}."
                    )

        # metric not available
        else:
            raise ValueError(f"Metric {metric} not available.")

    # Compute metrics ####################################################################

    # compute metric and return metric_df
    metric_df = pd.DataFrame(index=[0], columns=metric_kwargs.keys())
    for metric, kwargs in metric_kwargs.items():
        # get metric function
        metric_func = PROB_METRICS[metric]

        # compute metric value
        try:
            metric_df.loc[0, metric] = metric_func(**kwargs)
        except Exception as e:
            warnings.warn(f"Could not compute metric '{metric}': {e}", UserWarning)

            if return_on_error == "error":
                metric_df.loc[0, metric] = e
            else:
                metric_df.loc[0, metric] = return_on_error

        # select band name
        try:
            band_name = kwargs["member_dim"]  # member_dim passed as band name
        except KeyError:
            try:
                band_name = DEFAULT_METRIC_KWARGS[metric][
                    "member_dim"
                ]  # use default member_dim as band name
            except KeyError:
                band_name = "1"  # no member_dim passed and no default member_dim, use "1" as band name

        # add band column
        metric_df.insert(0, "band", band_name)

    # convert metric_dict to Prob_metrics_df
    metric_df = Prob_metrics_df(metric_df)

    return metric_df
