"""
Test functionality for computing_categorical_metrics.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"


import pandas as pd
from pytest_cases import parametrize_with_cases
from pytest import raises

from gval.comparison.compute_categorical_metrics import _compute_categorical_metrics


@parametrize_with_cases(
    "crosstab_df, metrics, positive_categories, negative_categories, expected_df",
    glob="compute_categorical_metrics_success",
)
def test_compute_categorical_metrics_success(
    crosstab_df, metrics, positive_categories, negative_categories, expected_df
):
    """tests categorical metrics functions"""

    # compute categorical metrics
    metrics_df = _compute_categorical_metrics(
        crosstab_df=crosstab_df,
        positive_categories=positive_categories,
        metrics=metrics,
        negative_categories=negative_categories,
    )

    pd.testing.assert_frame_equal(
        metrics_df, expected_df, check_dtype=False
    ), "Compute statistics did not return expected values"


@parametrize_with_cases(
    "crosstab_df, positive_categories, negative_categories, exception",
    glob="compute_categorical_metrics_fail",
)
def test_compute_categorical_metrics_fail(
    crosstab_df, positive_categories, negative_categories, exception
):
    """tests categorical metrics functions"""

    with raises(exception):
        # compute categorical metrics
        _compute_categorical_metrics(
            crosstab_df, "all", positive_categories, negative_categories
        )


@parametrize_with_cases(
    "crosstab_df, metrics, positive_categories, negative_categories, average, weights, expected_df",
    glob="compute_multi_categorical_metrics_success",
)
def test_compute_multi_categorical_metrics_success(
    crosstab_df,
    metrics,
    positive_categories,
    negative_categories,
    average,
    weights,
    expected_df,
):
    """tests multiclass categorical metrics functions"""

    # compute categorical metrics
    metrics_df = _compute_categorical_metrics(
        crosstab_df=crosstab_df,
        positive_categories=positive_categories,
        metrics=metrics,
        negative_categories=negative_categories,
        average=average,
        weights=weights,
    )

    pd.testing.assert_frame_equal(
        metrics_df, expected_df, check_dtype=False
    ), "Compute statistics did not return expected values"


@parametrize_with_cases(
    "crosstab_df, metrics, positive_categories, negative_categories, average, weights, exception",
    glob="compute_multi_categorical_metrics_fail",
)
def test_compute_multi_categorical_metrics_fail(
    crosstab_df,
    metrics,
    positive_categories,
    negative_categories,
    average,
    weights,
    exception,
):
    """tests multiclass categorical metrics functions"""

    with raises(exception):
        # compute categorical metrics
        _compute_categorical_metrics(
            crosstab_df=crosstab_df,
            positive_categories=positive_categories,
            metrics=metrics,
            negative_categories=negative_categories,
            average=average,
            weights=weights,
        )
