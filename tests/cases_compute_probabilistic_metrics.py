"""
Test functionality for compute_probabilistic_metrics.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from gval.utils.schemas import Prob_metrics_df
from gval.utils.loading_datasets import _create_xarray_pairs

nodata_value = np.nan
encoded_nodata_value = -9999
upper_left = (-95.2, 37.2)
lower_right = (-94.8, 36.8)
shapes = "circle"
return_datasets = False
sizes = 50
band_dim_name = "member"


def case_compute_prob_metrics_brier_score_success():
    expected_xr = xr.DataArray(0.22347608, coords={"spatial_ref": 0})

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"brier_score": {"member_dim": "member", "keep_attrs": True}},
        "return_on_error": "error",
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(
        pd.DataFrame({"band": ["member"], "brier_score": [None]})
    )
    expected_df.loc[0, "brier_score"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 1, (7, 7), 3),  # Band 1
        (0, 0, (12, 12), 5),  # Band 2
        (1, 1, (17, 17), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (8, 8), 4),  # Band 1
        (0, 0, (13, 13), 6),  # Band 2
        (1, 1, (18, 18), 8),  # Band 3
    ]

    # generate xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_crps_ensemble_success():
    return_datasets = False
    sizes = (50, 60)
    expected_xr = xr.DataArray(44.65896392, coords={"spatial_ref": 0})

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {
            "crps_ensemble": {"member_dim": "member", "keep_attrs": True}
        },
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(
        pd.DataFrame({"band": ["member"], "crps_ensemble": [None]})
    )
    expected_df.loc[0, "crps_ensemble"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (100, 300, (20, 20), 7),
        (150, 350, (30, 30), 10),
        (200, 400, (40, 40), 8),
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (145, 360, (25, 25), 15),
        # (160, 360, (2, 2), 3),
        # (210, 410, (3, 3), 4)
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    # drop member from benchmark_map
    benchmark_map = benchmark_map.squeeze().drop_vars("member")

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_crps_gaussian_success():
    expected_xr = xr.DataArray(209.8235419, coords={"spatial_ref": 0})
    band_dim_name = "band"

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"crps_gaussian": {"mu": 0, "sig": 1, "keep_attrs": False}},
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(
        pd.DataFrame({"band": ["1"], "crps_gaussian": [None]})
    )
    expected_df.loc[0, "crps_gaussian"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (140, 340, (9, 9), 3),  # Band 1
        (190, 390, (14, 14), 5),  # Band 2
        (240, 440, (19, 19), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (150, 350, (10, 10), 4),  # Band 1
        (200, 400, (15, 15), 6),  # Band 2
        (250, 450, (20, 20), 8),  # Band 3
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_crps_quadrature_success():
    return_datasets = True
    sizes = 5
    expected_xr = xr.Dataset(
        data_vars={"variable": 271.93581042}, coords={"spatial_ref": 0}
    )
    band_dim_name = "band"

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {
            "crps_quadrature": {"cdf_or_dist": norm, "tol": 1e-6, "keep_attrs": False}
        },
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(
        pd.DataFrame({"band": ["1"], "crps_quadrature": [None]})
    )
    expected_df.loc[0, "crps_quadrature"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (140, 340, (3, 3), 1),  # Band 1
        (190, 390, (2, 2), 1),  # Band 2
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (150, 350, (3, 3), 1),  # Band 1
        (200, 400, (2, 2), 1),  # Band 2
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_discrimination_success():
    expected_xr = xr.DataArray(
        data=np.empty((2, 0)),
        dims=["event", "forecast_probability"],
        coords={
            "event": [True, False],
            "spatial_ref": 0,
            "forecast_probability": np.array([], dtype="float64"),
        },
    )
    band_dim_name = "band"

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"discrimination": {"probability_bin_edges": np.array([200])}},
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(
        pd.DataFrame({"band": ["1"], "discrimination": [None]})
    )
    expected_df.loc[0, "discrimination"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (140, 340, (9, 9), 3),  # Band 1
        (190, 390, (14, 14), 5),  # Band 2
        (240, 440, (19, 19), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (150, 350, (10, 10), 4),  # Band 1
        (200, 400, (15, 15), 6),  # Band 2
        (250, 450, (20, 20), 8),  # Band 3
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_rank_histogram_success():
    expected_xr = xr.DataArray(
        data=np.array([3, 2252, 16, 33]),
        dims=["rank"],
        coords={"rank": np.array([1, 2, 3, 4])},
    )

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"rank_histogram": None},
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(
        pd.DataFrame({"band": ["member"], "rank_histogram": [None]})
    )
    expected_df.loc[0, "rank_histogram"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (140, 340, (9, 9), 3),  # Band 1
        (190, 390, (14, 14), 5),  # Band 2
        (240, 440, (19, 19), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (150, 350, (10, 10), 4),  # Band 1
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    # drop band from benchmark_map
    benchmark_map = benchmark_map.squeeze().drop_vars("member")

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_reliability_success():
    nodata_value = -1
    encoded_nodata_value = None

    expected_xr = xr.DataArray(
        data=np.array([0.35257523, np.nan, np.nan, np.nan, np.nan, np.nan]),
        dims=["forecast_probability"],
        coords={
            "forecast_probability": np.array([0.05, 0.2, 0.4, 0.6, 0.75, 0.85]),
            "samples": ("forecast_probability", np.array([6912, 0, 0, 0, 0, 0])),
            "spatial_ref": 0,
        },
    )
    band_dim_name = "band"

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {
            "reliability": {
                "probability_bin_edges": np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9])
            }
        },
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(pd.DataFrame({"band": ["1"], "reliability": [None]}))
    expected_df.loc[0, "reliability"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0.03, 0.76, (9, 9), 3),  # Band 1
        (0.89, 0.09, (14, 14), 5),  # Band 2
        (0.29, 0.63, (219, 19), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (10, 10), 4),  # Band 1
        (1, 0, (15, 15), 6),  # Band 2
        (0, 1, (20, 20), 8),  # Band 3
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_roc_success():
    nodata_value = 0
    encoded_nodata_value = None

    expected_xr = xr.DataArray(0.97954505, coords={"spatial_ref": 0})
    band_dim_name = "band"

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"roc": {"bin_edges": "continuous"}},
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(pd.DataFrame({"band": ["1"], "roc": [None]}))
    expected_df.loc[0, "roc"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 1, (9, 9), 3),  # Band 1
        (1, 0, (14, 14), 5),  # Band 2
        (0, 1, (19, 19), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (10, 10), 4),  # Band 1
        (1, 0, (15, 15), 6),  # Band 2
        (0, 1, (20, 20), 8),  # Band 3
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_rps_success():
    categories = np.array(
        "[-np.inf, 0.1), [0.1, 0.2), [0.2, 0.3), [0.3, 0.4), [0.4, 0.5), [0.5, 0.6), [0.6, np.inf]",
        dtype="<U89",
    )

    expected_xr = xr.DataArray(
        data=1.22604444,
        coords={
            "spatial_ref": 0,
            "observations_category_edge": categories,
            "forecasts_category_edge": categories,
        },
    )

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {
            "rps": {
                "category_edges": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                "member_dim": "member",
            }
        },
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(pd.DataFrame({"band": ["member"], "rps": [None]}))
    expected_df.loc[0, "rps"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0.08, 0.92, (9, 9), 3),  # Band 1
        (0.76, 0.1, (14, 14), 5),  # Band 2
        (0.12, 0.88, (19, 19), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0.2, 0.94, (10, 10), 4),  # Band 1
        (0.71, 0.13, (15, 15), 6),  # Band 2
        (0.19, 0.81, (20, 20), 8),  # Band 3
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_threshold_brier_score_success():
    expected_xr = xr.DataArray(
        data=np.array([0.133873, 0.133873, 0.133873, 0.133873, 0.133873, 0.133873]),
        dims=["threshold"],
        coords={
            "spatial_ref": 0,
            "threshold": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        },
    )

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {
            "threshold_brier_score": {"threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
        },
        "return_on_error": None,
    }

    # creating expected df. expected_xr is added later due to pd dtype issues
    expected_df = Prob_metrics_df(
        pd.DataFrame({"band": ["member"], "threshold_brier_score": [None]})
    )
    expected_df.loc[0, "threshold_brier_score"] = expected_xr

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 0.85, (9, 9), 3),  # Band 1
        (0.75, 0, (14, 14), 5),  # Band 2
        (0, 0.78, (19, 19), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (10, 10), 4),  # Band 1
    ]

    # Generate the xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    # drop band from benchmark_map
    benchmark_map = benchmark_map.squeeze().drop_vars("member")

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_error_success():
    expected_error = AttributeError

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"crps_gaussian": {"mu": 0, "sig": None}},
        "return_on_error": "error",
    }

    expected_df = Prob_metrics_df(
        pd.DataFrame({"band": ["1"], "crps_gaussian": [None]})
    )
    expected_df.loc[0, "crps_gaussian"] = expected_error

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (140, 340, (9, 9), 3),  # Band 1
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (150, 350, (10, 10), 4),  # Band 1
    ]

    # generate xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        None,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_df


def case_compute_prob_metrics_kwargs_required_fail():
    expected_error = ValueError

    # compute_kwargs
    compute_kwargs = {"metric_kwargs": {"crps_gaussian": None}, "return_on_error": None}

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 1, (7, 7), 3),  # Band 1
        (0, 0, (12, 12), 5),  # Band 2
        (1, 1, (17, 17), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (8, 8), 4),  # Band 1
        (0, 0, (13, 13), 6),  # Band 2
        (1, 1, (18, 18), 8),  # Band 3
    ]

    # generate xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        None,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_error


def case_compute_prob_metrics_specific_kwargs_required_fail():
    expected_error = ValueError

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"crps_gaussian": {"mu": 0}},
        "return_on_error": None,
    }

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 1, (7, 7), 3),  # Band 1
        (0, 0, (12, 12), 5),  # Band 2
        (1, 1, (17, 17), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (8, 8), 4),  # Band 1
        (0, 0, (13, 13), 6),  # Band 2
        (1, 1, (18, 18), 8),  # Band 3
    ]

    # generate xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        None,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_error


def case_compute_prob_metrics_kwargs_extra_fail():
    expected_error = ValueError

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"crps_gaussian": {"mu": None, "sig": None, "mu_sigma": 0}},
        "return_on_error": None,
    }

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 1, (7, 7), 3),  # Band 1
        (0, 0, (12, 12), 5),  # Band 2
        (1, 1, (17, 17), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (8, 8), 4),  # Band 1
        (0, 0, (13, 13), 6),  # Band 2
        (1, 1, (18, 18), 8),  # Band 3
    ]

    # generate xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        None,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_error


def case_compute_prob_metrics_missing_metric_fail():
    expected_error = ValueError

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"crps_test": {"mu": 0}},
        "return_on_error": None,
    }

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 1, (7, 7), 3),  # Band 1
        (0, 0, (12, 12), 5),  # Band 2
        (1, 1, (17, 17), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (8, 8), 4),  # Band 1
        (0, 0, (13, 13), 6),  # Band 2
        (1, 1, (18, 18), 8),  # Band 3
    ]

    # generate xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        None,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_error


def case_compute_prob_metrics_metric_func_warns():
    expected_warning = UserWarning

    # compute_kwargs
    compute_kwargs = {
        "metric_kwargs": {"crps_gaussian": {"mu": None, "sig": None}},
        "return_on_error": None,
    }

    # Band parameters for candidate
    # background value, circle value, circle center, and circle radius
    band_params_candidate = [
        (0, 1, (7, 7), 3),  # Band 1
        (0, 0, (12, 12), 5),  # Band 2
        (1, 1, (17, 17), 7),  # Band 3
    ]

    # Band parameters for benchmark
    # background value, circle value, circle center, and circle radius
    band_params_benchmark = [
        (0, 1, (8, 8), 4),  # Band 1
        (0, 0, (13, 13), 6),  # Band 2
        (1, 1, (18, 18), 8),  # Band 3
    ]

    # generate xarray pairs
    candidate_map, benchmark_map = _create_xarray_pairs(
        upper_left,
        lower_right,
        sizes,
        band_params_candidate,
        band_params_benchmark,
        None,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_datasets,
    )

    return candidate_map, benchmark_map, compute_kwargs, expected_warning
