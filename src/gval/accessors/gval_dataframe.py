from numbers import Number
from typing import Union, Iterable, Optional

import pandas as pd
from pandera.typing import DataFrame

from gval.comparison.compute_categorical_metrics import _compute_categorical_metrics
from gval.utils.schemas import Metrics_df


@pd.api.extensions.register_dataframe_accessor("gval")
class GVALDataFrame:
    """
    Class for extending pandas DataFrame functionality

    Attributes
    ----------
    _obj : pd.DataFrame
        Object to use off the accessor
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def compute_categorical_metrics(
        self,
        positive_categories: Union[Number, Iterable[Number]],
        negative_categories: Union[Number, Iterable[Number]],
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
        negative_categories : Optional[Union[Number, Iterable[Number]]], default = None
            Number or list of numbers representing the values to consider as the negative condition. This should be set to None when no negative categories are used or when the average type is "macro" or "weighted".
        metrics : Union[str, Iterable[str]], default = "all"
            String or list of strings representing metrics to compute.
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

        return _compute_categorical_metrics(
            crosstab_df=self._obj,
            metrics=metrics,
            positive_categories=positive_categories,
            negative_categories=negative_categories,
            average=average,
            weights=weights,
        )
