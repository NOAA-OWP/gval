from numbers import Number
from typing import Union, Iterable, Optional, List, Dict

import pandas as pd
from shapely import Geometry
import geopandas as gpd
from pandera.typing import DataFrame
import xarray as xr

from gval.comparison.compute_categorical_metrics import _compute_categorical_metrics
from gval.utils.schemas import Metrics_df, SubsamplingDf
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

    def compute_categorical_metrics(
        self,
        positive_categories: Union[
            Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]
        ],
        negative_categories: Union[
            Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]
        ],
        metrics: Union[str, Iterable[str]] = "all",
        average: str = "micro",
        weights: Optional[Iterable[Number]] = None,
        subsampling_average: Optional[str] = None,
    ) -> DataFrame[Metrics_df]:
        """
        Computes categorical metrics from a crosstab df.

        Parameters
        ----------
        crosstab_df : DataFrame[Crosstab_df]
            Crosstab DataFrame with candidate, benchmark, and agreement values as well as the counts for each occurrence.
        positive_categories : Optional[Union[Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]]]
            Number or list of numbers representing the values to consider as the positive condition. For average types "macro" and "weighted", this represents the categories to compute metrics for.
        negative_categories : Optional[UUnion[Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]]], default = None
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
        subsampling_average: Optional[str], default = None
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

        return _compute_categorical_metrics(
            crosstab_df=self._obj,
            metrics=metrics,
            positive_categories=positive_categories,
            negative_categories=negative_categories,
            average=average,
            weights=weights,
            sampling_average=subsampling_average,
        )

    def rasterize_data(
        self, reference_map: Union[xr.Dataset, xr.DataArray], rasterize_attributes: list
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Convenience function for rasterizing vector data using a reference raster.  For more control use `make_geocube`
        from the geocube package.

        Parameters
        ----------
        reference_map: Union[xr.Dataset, xr.DataArray]
            Map to reference in creation of rasterized vector map
        rasterize_attributes: list
            Attributes to rasterize

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            Rasterized Data

        Raises
        ------
        KeyError

        References
        ----------
        .. [1] `make_geocube <https://corteva.github.io/geocube/html/geocube.html>`_

        """

        return _rasterize_data(
            candidate_map=reference_map,
            benchmark_map=self._obj,
            rasterize_attributes=rasterize_attributes,
        )

    def create_subsampling_df(
        self,
        geometries: List[Geometry] = None,
        crs: str = None,
        subsampling_type: Union[str, List[str]] = "exclude",
        subsampling_weights: List[Union[int, float]] = None,
        inplace: bool = False,
    ) -> Union[None, SubsamplingDf]:
        """
        Parameters
        __________
        geometries: List[Geometry], default = None
            Geometries if none are already in the GeoDataFrame
        crs: str
            The spatial reference for the geometries provided
        subsampling_type: Union[str, List[str]], default = "exclude"
            Whether each geometry should be an inclusive subsample or an exclusionary mask
        subsampling_weights: List[Union[int, float]], default = None
            Values to scale the numeric impact of a particular sample
        inplace: bool, default = False
            Whether to adjust the GeoDataFrame calling the operation or a return a new one

        Raises
        ------
        ValueError
            List provided has more or less entries than the DataFrame
        TypeError
            CRS must be provided if geometries are provided

        Returns
        -------
        Union[None, SubsamplingDf]
            GeoDataFrame adhering to subsampling dataframe if not inplace, otherwise None

        """

        geo_df = self._obj if inplace else gpd.GeoDataFrame()
        crs = crs if crs is not None else self._obj.crs

        if (geometries and inplace) or ("geometry" in self._obj.columns):
            if crs is None:
                raise TypeError(
                    "Either provide CRS for or give the original DataFrame a crs if inplace is True"
                )  # pragma: no cover

            if "geometry" not in geo_df.columns:
                geo_df.set_geometry(
                    geometries if geometries is not None else self._obj["geometry"],
                    inplace=True,
                )
                geo_df.crs = crs

        if subsampling_type:
            geo_df["subsample_type"] = subsampling_type

        if subsampling_weights:
            geo_df["weights"] = subsampling_weights

        geo_df.loc[:, "subsample_id"] = geo_df.index.values + 1

        if not inplace:
            return geo_df
