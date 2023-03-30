from numbers import Number
from typing import Union, Iterable, Optional

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
        metrics: Union[str, Iterable[str]] = "all",
        positive_categories: Optional[Union[Number, Iterable[Number]]] = None,
        negative_categories: Optional[Union[Number, Iterable[Number]]] = None,
    ) -> DataFrame[Metrics_df]:
        """

        Parameters
        ----------
        metrics: Union[str, Iterable[str]], default = "all"
            Statistics to return in metric table
        positive_categories: Optional[Union[Number, Iterable[Number]]], default = None
            Categories to represent positive entries
        negative_categories: Optional[Union[Number, Iterable[Number]]], default = None
            Categories to represent negative entries

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
