"""
Computes categorical metrics given a crosstab df.
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Iterable, Optional, Union, Dict
from numbers import Number

# deprecation warnings: DeprecationWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array. To retain the old behavior, use either `df[df.columns[i]] = newvals` or, if columns are non-unique, `df.isetitem(i, newvals)
import warnings

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from gval import CatStats
from gval.utils.schemas import (
    Crosstab_df,
    Sample_identifiers,
    Subsample_identifiers,
    Metrics_df,
)


def _create_set_dict(categories):
    """Creates set valued dictionary"""
    if not isinstance(categories, dict):
        categories = {"candidate": categories, "benchmark": categories}
    else:
        if "candidate" not in categories or "benchmark" not in categories:
            raise ValueError(
                "Make sure both candidate and benchmark entries are in the dictionary passed"
                "in to negative_categories, positive_categories, or both."
            )

    categories["candidate"] = _convert_to_set(categories["candidate"])
    categories["benchmark"] = _convert_to_set(categories["benchmark"])

    return categories


def _convert_to_set(data):
    """Converts data to set"""
    if data is None:
        data = set()
    else:
        data = {data} if isinstance(data, Number) else set(data)

    return data


def _get_unique_values(crosstab_df, column):
    """Get unique values from a Dataframe"""

    return set(
        [
            item
            for item in crosstab_df.loc[:, column].to_numpy().ravel()
            if not isinstance(item, list)
        ]
    )


def _handle_positive_negative_categories(
    crosstab_df, positive_categories, negative_categories
):  # pragma: no cover
    """Input handling for use case with positive and negative categories"""

    positive_categories = _create_set_dict(positive_categories)
    negative_categories = _create_set_dict(negative_categories)

    # input check to make sure the same value isn't repeated in positive and negative categories
    if positive_categories["candidate"].intersection(negative_categories["candidate"]):
        raise ValueError(
            "Value is shared in candidate positive and negative categories."
        )

    if positive_categories["benchmark"].intersection(negative_categories["benchmark"]):
        raise ValueError(
            "Value is shared in benchmark positive and negative categories."
        )

    # Finds the unique values in the sample's candidate and benchmark values
    candidate_unique_values = _get_unique_values(crosstab_df, "candidate_values")
    benchmark_unique_values = _get_unique_values(crosstab_df, "benchmark_values")

    # Share unique values if both candidate and benchmark share the same categories
    if (
        positive_categories["candidate"] == positive_categories["benchmark"]
        and negative_categories["candidate"] == negative_categories["benchmark"]
    ):
        candidate_unique_values = (
            benchmark_unique_values
        ) = candidate_unique_values.union(benchmark_unique_values)

    # This checks that user passed positive or negative categories exist in sample df
    def categories_exist_in_crosstab_df(categories, cat_name, unique_values):
        for c in categories:
            if c not in unique_values:
                raise ValueError(f"{cat_name} category {c} not found in crosstab df.")

    categories_exist_in_crosstab_df(
        positive_categories["candidate"], "positive", candidate_unique_values
    )
    categories_exist_in_crosstab_df(
        negative_categories["candidate"], "negative", candidate_unique_values
    )
    categories_exist_in_crosstab_df(
        positive_categories["benchmark"], "positive", benchmark_unique_values
    )
    categories_exist_in_crosstab_df(
        negative_categories["benchmark"], "negative", benchmark_unique_values
    )

    return positive_categories, negative_categories


@pa.check_types
def _compute_categorical_metrics(
    crosstab_df: DataFrame[Crosstab_df],
    positive_categories: Optional[
        Union[Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]]
    ],
    negative_categories: Optional[
        Union[Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]]
    ] = None,
    metrics: Union[str, Iterable[str]] = "all",
    average: str = "micro",
    weights: Optional[Iterable[Number]] = None,
    sampling_average: Optional[str] = None,
) -> DataFrame[Metrics_df]:
    """
    Computes categorical metrics from a crosstab df.

    Parameters
    ----------
    crosstab_df : DataFrame[Crosstab_df]
        Crosstab DataFrame with candidate, benchmark, and agreement values as well as the counts for each occurrence.
    positive_categories : Optional[Union[Number, Iterable[Number]]]
        Number or list of numbers representing the values to consider as the positive condition. For average types "macro" and "weighted", this represents the categories to compute metrics for.
    metrics : Union[str, Iterable[str]], default = "all"
        String or list of strings representing metrics to compute.
    negative_categories : Optional[Union[Number, Iterable[Number]]], default = None
        Number or list of numbers representing the values to consider as the negative condition. This should be set to None when no negative categories are used or when the average type is "macro" or "weighted".
    average : str, default = "micro"
        Type of average to use when computing metrics. Options are "micro", "macro", and "weighted".
        Micro weighing computes the conditions, tp, tn, fp, and fn, for each category and then sums them.
        Macro weighing computes the metrics for each category then averages them.
        Weighted average computes the metrics for each category then averages them weighted by the number of weights argument in each category.
    weights : Optional[Iterable[Number]], default = None
        Weights to use when computing weighted average. Elements correspond to positive categories in order.

        Example:

        `positive_categories = [1, 2]; weights = [0.25, 0.75]`
    sampling_average: Optional[str], default = None
        Way to aggregate statistics for subsamples if provided. Options are "sample", "band", and "full-detail"
        Sample calculates metrics and averages the results by subsample
        Band calculates metrics and averages all the metrics by band
        Full-detail does not aggregation on subsample or band

    Returns
    -------
    DataFrame[Metrics_df]
        Metrics DF with computed metrics per sample.

    Raises
    ------
    ValueError
        Value is shared in positive and negative categories.
    ValueError
        Category not found in crosstab df.
    ValueError
        Cannot use average type with only one positive category.
    ValueError
        Number of weights must be the same as the number of positive categories.
    ValueError
        Cannot use average type with negative_categories as not None. Set negative_categories to None for this average type.

    References
    ----------
    .. [1] `Evaluation of binary classifiers <https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers>`_
    .. [2] `7th International Verification Methods Workshop <https://www.cawcr.gov.au/projects/verification/#Contingency_table>`_
    .. [3] `3.3. Metrics and scoring: quantifying the quality of predictions <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
    """

    #########################################################################################
    # input handling

    # make copy to avoid modifying original
    crosstab_df = crosstab_df.copy()

    positive_categories, negative_categories = _handle_positive_negative_categories(
        crosstab_df, positive_categories, negative_categories
    )

    # check to make sure that macro and weighted average are not used with only one category
    if average in ["macro", "weighted"] and len(positive_categories["candidate"]) == 1:
        raise ValueError(
            f"Cannot use average type '{average}' with only one positive category."
        )

    # check to make sure length of weights is the same as positive categories
    if average == "weighted":
        if len(weights) != len(positive_categories["candidate"]):
            raise ValueError(
                f"Number of positive category weights ({len(weights)}) must be the same as the number of positive "
                f"categories ({len(positive_categories['candidate'])})."
            )

    # make sure negative categories are not used with macro or weighted average
    if average in ["macro", "weighted"] and negative_categories["candidate"]:
        raise ValueError(
            f"Cannot use average type '{average}' with negative_categories as not None. "
            f"Set negative_categories to None for this average type."
        )

    if (
        average in ["macro", "weighted"]
        and positive_categories["candidate"] != positive_categories["benchmark"]
    ):
        raise ValueError(
            f"Cannot use average type '{average}' with different classes in positive candidate and benchmark values. "
            f"Make sure they are the same."
        )

    #########################################################################################
    # assign conditions

    def assign_condition_to_pairing(row, positive_categories, negative_categories):
        """This is used to create a conditions column"""
        # predicted positive
        if row["candidate_values"] in positive_categories["candidate"]:
            if row["benchmark_values"] in positive_categories["benchmark"]:
                return "tp"
            elif (
                row["benchmark_values"] in negative_categories["benchmark"]
                or len(negative_categories["benchmark"]) == 0
            ):
                return "fp"

        # predicted negative
        elif (
            row["candidate_values"] in negative_categories["candidate"]
            or len(negative_categories["candidate"]) == 0
        ):
            if row["benchmark_values"] in positive_categories["benchmark"]:
                return "fn"
            elif row["benchmark_values"] in negative_categories["benchmark"]:
                return "tn"

    #########################################################################################
    # compute metrics

    def compute_metrics_per_sample(sample):
        """Computes the metric values on a per sample basis"""

        # this turns the sample df to a groupby object on conditions
        conditions_series = (
            sample.loc[:, ["counts", "conditions"]]
            .groupby("conditions")
            .sum()[
                "counts"
            ]  # this is only necessary for micro average but left in place regardless
        )
        # .loc[Conditions_df.columns()] # used for ordering conditions

        # compute metric values and return names
        metric_values, metric_names = CatStats.process_statistics(
            metrics, **conditions_series.to_dict()
        )

        metrics_series = pd.Series(metric_values, index=metric_names)

        return pd.concat([conditions_series, metrics_series])

    #########################################################################################
    # computes metrics per category or micro level category

    def compute_metrics_per_category(pos_cats, neg_cats, groupby_cols):
        # assign conditions to crosstab_df
        crosstab_df["conditions"] = crosstab_df.apply(
            assign_condition_to_pairing, axis=1, args=(pos_cats, neg_cats)
        )

        # groupby sample identifiers then compute metrics
        metric_df = (
            crosstab_df.groupby(groupby_cols)
            .apply(compute_metrics_per_sample)
            .reset_index()
        )

        if "tn" not in metric_df.columns:
            metric_df.insert(
                loc=metric_df.columns.get_loc("tp"), column="tn", value=np.nan
            )

        # warnings for deprecation
        with warnings.catch_warnings(record=True):
            # Filter warnings by category, in this case, DeprecationWarning
            warnings.simplefilter("always", DeprecationWarning)
            if len(pos_cats["candidate"]) == 1:
                metric_df.insert(
                    loc=metric_df.columns.get_loc("fn"),
                    column="positive_categories",
                    value=None,
                )
                metric_df.loc[:, "positive_categories"] = pos_cats["candidate"].pop()
            elif len(metric_df) == len(pos_cats["candidate"]):
                metric_df.insert(
                    loc=metric_df.columns.get_loc("fn"),
                    column="positive_categories",
                    value=None,
                )
                metric_df.loc[:, "positive_categories"] = list(pos_cats)

        return metric_df

    #########################################################################################
    # Get subsampling_grouping
    if sampling_average == "band":
        groupby = Subsample_identifiers.columns()
    elif sampling_average == "subsample" or sampling_average is None:
        groupby = Sample_identifiers.columns()
    else:
        groupby = list(
            np.ravel([Sample_identifiers.columns(), Subsample_identifiers.columns()])
        )  # full-detail

    # replaces metrics variable with all metrics if metrics is set to all
    metrics = CatStats.available_functions() if metrics == "all" else metrics

    # actually compute metrics given averaging scheme
    if average == "micro":
        metric_df = compute_metrics_per_category(
            positive_categories, negative_categories, groupby
        )
    elif (average == "macro") or (average == "weighted") or (average is None):
        # Make copies to mutate
        positive_copy = positive_categories.copy()
        negative_copy = negative_categories.copy()
        metrics_dfs = []

        # One vs All Class Multi-categorical
        for pos_cat in positive_categories["candidate"]:
            positive_copy["candidate"] = positive_copy["benchmark"] = {pos_cat}
            negative_copy["candidate"] = negative_copy[
                "benchmark"
            ] = positive_categories["candidate"].difference({pos_cat})
            metrics_dfs.append(
                compute_metrics_per_category(positive_copy, negative_copy, groupby)
            )

        metric_df = pd.concat(metrics_dfs)

    #########################################################################################
    # aggregate metrics

    if average is None:
        metric_df.reset_index(drop=True, inplace=True)

    elif average == "micro":
        metric_df = metric_df.drop(
            columns=["positive_categories"], errors="ignore"
        ).reset_index(drop=True)

    elif average == "macro":
        metric_df = (
            metric_df.groupby(groupby)
            .mean(numeric_only=True)
            .drop(
                columns=["fn", "fp", "tn", "tp", "positive_categories"], errors="ignore"
            )
            .reset_index()
        )

    elif average == "weighted":
        if (weights is not None) & (average == "weighted"):
            # turn weights into a dict paired with positive categories
            weights = dict(zip(metric_df["positive_categories"], weights))

            def assign_weights(row):
                """This is used to create a weights column"""
                try:
                    return weights[row["positive_categories"]]
                except KeyError:
                    return np.nan

            # add a column in crosstab_df called weights
            metric_df.insert(
                loc=metric_df.columns.get_loc("positive_categories"),
                column="weights",
                value=np.nan,
            )

            metric_df["weights"] = metric_df.apply(assign_weights, axis=1)

        # TODO: WHERE ARE THE INDEXES NOT BEING RESET PREVIOUSLY?
        metric_df.reset_index(drop=True, inplace=True)

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
                columns=["fn", "fp", "tn", "tp", "weights", "positive_categories"],
                errors="ignore",
            )
            .divide(metric_df.loc[:, "weights"].sum())
            .reset_index()
        )

    if sampling_average == "band":
        metric_df.insert(1, "band", "averaged")
    elif sampling_average == "subsample":
        metric_df.insert(0, "subsample", "averaged")

    return metric_df
