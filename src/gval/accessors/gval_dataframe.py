from numbers import Number
from typing import Union, Iterable

import pandas as pd
from pandera.typing import DataFrame
import xarray as xr

from gval.comparison.compute_categorical_metrics import _compute_categorical_metrics
from gval.utils.schemas import Metrics_df
from gval.homogenize.rasterize import _rasterize_data


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

    def rasterize_data(
        self, target_map: Union[xr.DataArray, xr.Dataset], rasterize_attributes: list
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Method to rasterize a GeoDataFrame

        Parameters
        ----------
        target_map: Union[xr.DataArray, xr.Dataset]
            Target map to reference in rasterize process
        rasterize_attributes: list
            Numerical attributes to rasterize

        Returns
        -------
        Union[xr.DataArray, xr.Dataset]
            Rasterized xarray object
        """

        if "geometry" in self._obj.columns:
            return _rasterize_data(
                candidate_map=target_map,
                benchmark_map=self._obj,
                rasterize_attributes=rasterize_attributes,
            )
        else:
            raise TypeError("GeoDataFrame is needed to rasterize data")
