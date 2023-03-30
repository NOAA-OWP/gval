from numbers import Number
from typing import Union, Iterable

import pandas as pd
from pandera.typing import DataFrame

from gval.comparison.compute_categorical_metrics import _compute_categorical_metrics
from gval.utils.schemas import Metrics_df


@pd.api.extensions.register_dataframe_accessor("gval")
class GVALDataFrame:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def compute_metrics(
        self,
        positive_categories: Union[Number, Iterable[Number]],
        negative_categories: Union[Number, Iterable[Number]],
        metrics: Union[str, Iterable[str]] = "all",
    ) -> DataFrame[Metrics_df]:
        """

        Parameters
        ----------
        positive_categories: Union[Number, Iterable[Number]]
            Categories to represent positive entries
        negative_categories: Union[Number, Iterable[Number]]
            Categories to represent negative entries
        metrics: Union[str, Iterable[str]], default = "all"
            Statistics to return in metric table


        Returns
        -------
        DataFrame[Metrics_df]
            Metrics table
        """

        return _compute_categorical_metrics(
            crosstab_df=self._obj,
            metrics=metrics,
            positive_categories=positive_categories,
            negative_categories=negative_categories,
        )
