"""
Computes categorical metrics given a crosstab df.
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Iterable, Optional, Union
from numbers import Number

# deprecation warnings: DeprecationWarning: In a future version, `df.iloc[:, i] = newvals` will attempt to set the values inplace instead of always setting a new array. To retain the old behavior, use either `df[df.columns[i]] = newvals` or, if columns are non-unique, `df.isetitem(i, newvals)
import warnings

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from gval import CatStats
from gval.utils.schemas import Crosstab_df, Sample_identifiers, Metrics_df


def _handle_positive_negative_categories(
    crosstab_df, positive_categories, negative_categories
):  # pragma: no cover
    """Input handling for use case with positive and negative categories"""

    # convert to positive categories to sets
    if isinstance(positive_categories, Number):
        positive_categories = {positive_categories}
    else:
        positive_categories = set(positive_categories)

    # convert to negative categories to sets
    if isinstance(negative_categories, Number):
        negative_categories = {negative_categories}
    elif negative_categories is None:
        # TODO: Currently not working bc process_statistics requires all four conditions for "all" case
        negative_categories = set()
    else:
        negative_categories = set(negative_categories)

    # input check to make sure the same value isn't repeated in positive and negative categories
    if positive_categories.intersection(negative_categories):
        raise ValueError("Value is shared in positive and negative categories.")

    # finds the unique values in the sample's candidate and benchmark values
    unique_values = set(
        crosstab_df.loc[:, ["candidate_values", "benchmark_values"]].to_numpy().ravel()
    )

    # this checks that user passed positive or negative categories exist in sample df
    def categories_exist_in_crosstab_df(categories, cat_name):
        for c in categories:
            if c not in unique_values:
                raise ValueError(f"{cat_name} category {c} not found in crosstab df.")

    categories_exist_in_crosstab_df(positive_categories, "positive")
    categories_exist_in_crosstab_df(negative_categories, "negative")

    return positive_categories, negative_categories


@pa.check_types
def _compute_categorical_metrics(
    crosstab_df: DataFrame[Crosstab_df],
    positive_categories: Optional[Union[Number, Iterable[Number]]],
    negative_categories: Optional[Union[Number, Iterable[Number]]] = None,
    metrics: Union[str, Iterable[str]] = "all",
    average: str = "micro",
    weights: Optional[Iterable[Number]] = None,
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
    .. [1] [Evaluation of binary classifiers](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
    .. [2] [7th International Verification Methods Workshop](https://www.cawcr.gov.au/projects/verification/#Contingency_table)
    .. [3] [3.3. Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
    """

    #########################################################################################
    # input handling

    # make copy to avoid modifying original
    crosstab_df = crosstab_df.copy()

    positive_categories, negative_categories = _handle_positive_negative_categories(
        crosstab_df, positive_categories, negative_categories
    )

    # check to make sure that macro and weighted average are not used with only one category
    if average in ["macro", "weighted"] and len(positive_categories) == 1:
        raise ValueError(
            f"Cannot use average type '{average}' with only one positive category."
        )

    # check to make sure length of weights is the same as positive categories
    if average == "weighted":
        if len(weights) != len(positive_categories):
            raise ValueError(
                f"Number of weights ({len(weights)}) must be the same as the number of positive categories ({len(positive_categories)})."
            )

    # make sure negative categories are not used with macro or weighted average
    if average in ["macro", "weighted"] and negative_categories:
        raise ValueError(
            f"Cannot use average type '{average}' with negative_categories as not None. Set negative_categories to None for this average type."
        )

    #########################################################################################
    # assign conditions

    def assign_condition_to_pairing(row, positive_categories, negative_categories):
        """This is used to create a conditions column"""
        # predicted positive
        if row["candidate_values"] in positive_categories:
            if row["benchmark_values"] in positive_categories:
                return "tp"
            elif (
                row["benchmark_values"] in negative_categories
                or len(negative_categories) == 0
            ):
                return "fp"

        # predicted negative
        elif (
            row["candidate_values"] in negative_categories
            or len(negative_categories) == 0
        ):
            if row["benchmark_values"] in positive_categories:
                return "fn"
            elif row["benchmark_values"] in negative_categories:
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

    def compute_metrics_per_category(pos_cats, neg_cats):
        if not isinstance(pos_cats, set):
            pos_cats = set([pos_cats])

        # assign conditions to crosstab_df
        crosstab_df["conditions"] = crosstab_df.apply(
            assign_condition_to_pairing, axis=1, args=(pos_cats, neg_cats)
        )

        # groupby sample identifiers then compute metrics
        metric_df = (
            crosstab_df.groupby(Sample_identifiers.columns())
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
            if len(pos_cats) == 1:
                metric_df.insert(
                    loc=metric_df.columns.get_loc("fn"),
                    column="positive_categories",
                    value=None,
                )
                metric_df.loc[:, "positive_categories"] = pos_cats.pop()
            elif len(metric_df) == len(pos_cats):
                metric_df.insert(
                    loc=metric_df.columns.get_loc("fn"),
                    column="positive_categories",
                    value=None,
                )
                metric_df.loc[:, "positive_categories"] = list(pos_cats)

        return metric_df

    #########################################################################################
    # compute metrics

    # replaces metrics variable with all metrics if metrics is set to all
    metrics = CatStats.available_functions() if metrics == "all" else metrics

    # actually compute metrics given averaging scheme
    if average == "micro":
        metric_df = compute_metrics_per_category(
            positive_categories, negative_categories
        )
    elif (average == "macro") or (average == "weighted") or (average is None):
        metric_df = pd.concat(
            [
                compute_metrics_per_category(
                    pos_cat, positive_categories.difference({pos_cat})
                )
                for pos_cat in positive_categories
            ]
        )

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
            metric_df.groupby(Sample_identifiers.columns())
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

    return metric_df
