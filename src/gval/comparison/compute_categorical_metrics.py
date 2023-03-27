"""
Computes categorical metrics given a crosstab df.
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Iterable, Optional, Union
from numbers import Number

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
        raise ValueError("Value is shared in positive and negative ")

    # finds the unique values in the sample's candidate and benchmark values
    unique_values = set(
        crosstab_df.loc[:, ["candidate_values", "benchmark_values"]].to_numpy().ravel()
    )

    # this checks that user passed positive or negative categories exist in sample df
    def categories_exist_in_crosstab_df(categories, cat_name):
        for c in categories:
            if c not in unique_values:
                raise ValueError(
                    f"{cat_name} category {c} not found in crosstab df."
                )

    categories_exist_in_crosstab_df(positive_categories, "positive")
    categories_exist_in_crosstab_df(negative_categories, "negative")

    return positive_categories, negative_categories


@pa.check_types
def _compute_categorical_metrics(
    crosstab_df: DataFrame[Crosstab_df],
    metrics: Union[str, Iterable[str]] = "all",
    positive_categories: Optional[Union[Number, Iterable[Number]]] = None,
    negative_categories: Optional[Union[Number, Iterable[Number]]] = None,
    # candidate_categories: Optional[dict[str, Iterable[Number]]] = None,
    # benchmark_categories: Optional[dict[str, Iterable[Number]]] = None,
) -> DataFrame[Metrics_df]:
    """
    Computes categorical metrics from a crosstab df.

    Parameters
    ----------
    crosstab_df: pd.DataFrame

    positive_class: Number

    Returns
    -------
    pd.DataFrame
        Metric table.

    Raises
    ------

    References
    ----------
    .. [1] [Evaluation of binary classifiers](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
    """

    #########################################################################################
    # input handling

    # if (positive_categories is not None):
    positive_categories, negative_categories = _handle_positive_negative_categories(
        crosstab_df, positive_categories, negative_categories
    )

    #########################################################################################

    def assign_condition_to_pairing(row):
        """This is used to create a conditions column"""
        # predicted positive
        if row["candidate_values"] in positive_categories:
            if row["benchmark_values"] in positive_categories:
                return "tp"
            elif row["benchmark_values"] in negative_categories:
                return "fp"

        # predicted negative
        elif row["candidate_values"] in negative_categories:
            if row["benchmark_values"] in positive_categories:
                return "fn"
            elif row["benchmark_values"] in negative_categories:
                return "tn"

    # assign conditions to crosstab_df
    crosstab_df["conditions"] = crosstab_df.apply(assign_condition_to_pairing, axis=1)

    #########################################################################################
    # compute metrics

    def compute_metrics_per_sample(sample):
        """Computes the metric values on a per sample basis"""
        # this turns the sample df to a series
        conditions_series = (
            sample.loc[:, ["counts", "conditions"]]
            .groupby("conditions")
            .sum()["counts"]
        )  # \
        # .loc[Conditions_df.columns()] # used for ordering conditions

        # compute metric values and return names
        metric_values, metric_names = CatStats.process_statistics(
            metrics, **conditions_series.to_dict()
        )

        metrics_series = pd.Series(metric_values, index=metric_names)

        return pd.concat([conditions_series, metrics_series])

    # groupby sample identifiers then compute metrics
    return (
        crosstab_df.groupby(Sample_identifiers.columns())
        .apply(compute_metrics_per_sample)
        .reset_index()
    )
