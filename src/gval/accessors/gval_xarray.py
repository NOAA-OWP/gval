"""
Defines gval accessor for xarray objects.
"""

from typing import Iterable, Optional, Tuple, Union, Callable, Dict, List, Any
from numbers import Number

import numpy as np
import numba as nb
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling
from pandera.typing import DataFrame
import geopandas as gpd
import contextily as cx
import xyzservices

from gval.homogenize.spatial_alignment import _spatial_alignment
from gval.homogenize.rasterize import _rasterize_data
from gval.homogenize.vectorize import _vectorize_data
from gval.homogenize.numeric_alignment import _align_numeric_data_type
from gval import Comparison
from gval.comparison.tabulation import _crosstab_Datasets, _crosstab_DataArrays
from gval.comparison.compute_categorical_metrics import _compute_categorical_metrics
from gval.comparison.compute_continuous_metrics import _compute_continuous_metrics
from gval.comparison.compute_probabilistic_metrics import _compute_probabilistic_metrics
from gval.attributes.attributes import _attribute_tracking_xarray
from gval.utils.loading_datasets import _parse_string_attributes
from gval.utils.visualize import _map_plot
from gval.comparison.pairing_functions import difference
from gval.subsampling.subsampling import subsample
from gval.utils.schemas import (
    Crosstab_df,
    Metrics_df,
    AttributeTrackingDf,
    Prob_metrics_df,
)


class GVALXarray:
    """
    Class for extending xarray functionality.

    Attributes
    ----------
    _obj : Union[xr.Dataset, xr.DataArray]
       Object to use off the accessor
    data_type : type
       Data type of the _obj
    """

    def __init__(self, xarray_obj):
        self._obj = _parse_string_attributes(xarray_obj)
        self.data_type = type(xarray_obj)
        self.agreement_map_format = "raster"

    def check_same_type(self, benchmark_map: Union[xr.Dataset, xr.DataArray]):
        """
        Makes sure benchmark map is the same data type as the candidate object

        Parameters
        ----------
        benchmark_map: Union[xr.Dataset, xr.DataArray]
            Benchmark Map

        Raises
        ------
        TypeError

        """
        if not isinstance(benchmark_map, self.data_type):
            raise TypeError(f"Benchmark Map needs to be data type of {self.data_type}")

    def __handle_attribute_tracking(
        self,
        candidate_map: Union[xr.Dataset, xr.DataArray],
        benchmark_map: Union[xr.Dataset, xr.DataArray],
        agreement_map: Optional[Union[xr.Dataset, xr.DataArray]] = None,
        attribute_tracking_kwargs: Optional[Dict] = None,
    ):  # pragma: no cover
        """
        Handles attribute tracking for categorical and continuous comparison
        """

        # use user passed attribute_tracking_kwargs to pass arguments to attribute_tracking_xarray()
        if attribute_tracking_kwargs is not None:
            if "benchmark_map" in attribute_tracking_kwargs:
                del attribute_tracking_kwargs["benchmark_map"]

            if "agreement_map" in attribute_tracking_kwargs:
                if attribute_tracking_kwargs["agreement_map"] is None:
                    agreement_map = None
                else:
                    del attribute_tracking_kwargs["agreement_map"]

            results = _attribute_tracking_xarray(
                candidate_map=candidate_map,
                benchmark_map=benchmark_map,
                agreement_map=agreement_map,
                **attribute_tracking_kwargs,
            )

        else:
            results = _attribute_tracking_xarray(
                candidate_map=candidate_map,
                benchmark_map=benchmark_map,
                agreement_map=agreement_map,
            )

        return results

    @Comparison.comparison_function_from_string
    def categorical_compare(
        self,
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray],
        positive_categories: Union[
            Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]
        ],
        comparison_function: Union[
            Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str
        ] = "szudzik",
        metrics: Union[str, Iterable[str]] = "all",
        target_map: Optional[Union[xr.Dataset, str]] = "benchmark",
        resampling: Optional[Resampling] = Resampling.nearest,
        pairing_dict: Optional[Dict[Tuple[Number, Number], Number]] = None,
        allow_candidate_values: Optional[Iterable[Union[int, float]]] = None,
        allow_benchmark_values: Optional[Iterable[Union[int, float]]] = None,
        nodata: Optional[Number] = None,
        encode_nodata: Optional[bool] = False,
        exclude_value: Optional[Number] = None,
        negative_categories: Union[
            Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]
        ] = None,
        average: str = "micro",
        weights: Optional[Iterable[Number]] = None,
        rasterize_attributes: Optional[list] = None,
        attribute_tracking: bool = False,
        attribute_tracking_kwargs: Optional[Dict] = None,
        subsampling_df: Optional[gpd.GeoDataFrame] = None,
        subsampling_average: Optional[str] = None,
    ) -> Tuple[
        Union[
            Tuple[
                Union[xr.Dataset, xr.DataArray],
                DataFrame[Crosstab_df],
                DataFrame[Metrics_df],
            ],
            Tuple[
                Union[xr.Dataset, xr.DataArray],
                DataFrame[Crosstab_df],
                DataFrame[Metrics_df],
                DataFrame[AttributeTrackingDf],
            ],
        ]
    ]:
        """
        Computes comparison between two categorical value xarray's.

        Conducts the following steps:
            - homogenize: aligns data types, spatial alignment, and rasterizes data
            - compute_agreement: computes agreement map
            - compute_crosstab: computes crosstabulation
            - compute_metrics: computes metrics

        Spatially aligning the xarray's produces copies of the original candidate and benchmark maps. To reduce memory usage, consider using the `homogenize()` accessor method to overwrite the original maps in memory or saving them on disk.

        Parameters
        ----------
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray]
            Benchmark map.
        positive_categories: Union[Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]], default = None
            Number or list of numbers representing the values to consider as the positive condition. When the average argument is either "macro" or "weighted", this represents the categories to compute metrics for.
        comparison_function : Union[Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str], default = 'szudzik'
            Comparison function. Created by decorating function with @nb.vectorize() or using np.ufunc(). Use of numba is preferred as it is faster. Strings with registered comparison_functions are also accepted. Possible options include "pairing_dict". If passing "pairing_dict" value, please see the description for the argument for more information on behaviour.
            All available comparison functions can be found with gval.Comparison.available_functions().
        metrics: Union[str, Iterable[str]], default = "all"
            Statistics to return in metric table.  All returns every default and registered metric.  This can be seen with gval.CatStats.available_functions().
        target_map: Optional[Union[xr.Dataset, str]], default = "benchmark"
            xarray object to match the CRS's and coordinates of candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
        resampling : rasterio.enums.Resampling
            See :func:`rasterio.warp.reproject` for more details.
        pairing_dict: Optional[Dict[Tuple[Number, Number], Number]], default = None
            When "pairing_dict" is used for the comparison_function argument, a pairing dictionary can be passed by user. A pairing dictionary is structured as `{(c, b) : a}` where `(c, b)` is a tuple of the candidate and benchmark value pairing, respectively, and `a` is the value for the agreement array to be used for this pairing.

            If None is passed for pairing_dict, the allow_candidate_values and allow_benchmark_values arguments are required. For this case, the pairings in these two iterables will be paired in the order provided and an agreement value will be assigned to each pairing starting with 0 and ending with the number of possible pairings.

            A pairing dictionary can be used by the user to note which values to allow and which to ignore for comparisons. It can also be used to decide how nans are handled for cases where either the candidate and benchmark maps have nans or both.
        allow_candidate_values : Optional[Iterable[Union[int,float]]], default = None
            List of values in candidate to include in computation of agreement map. Remaining values are excluded. If "pairing_dict" is provided for comparison_function and pairing_function is None, this argument is necessary to construct the dictionary. Otherwise, this argument is optional and by default this value is set to None and all values are considered.
        allow_benchmark_values : Optional[Iterable[Union[int,float]]], default = None
            List of values in benchmark to include in computation of agreement map. Remaining values are excluded. If "pairing_dict" is provided for comparison_function and pairing_function is None, this argument is necessary to construct the dictionary. Otherwise, this argument is optional and by default this value is set to None and all values are considered.
        nodata : Optional[Number], default = None
            No data value to write to agreement map output. This will use `rxr.rio.write_nodata(nodata)`.
        encode_nodata : Optional[bool], default = False
            Encoded no data value to write to agreement map output. A nodata argument must be passed. This will use `rxr.rio.write_nodata(nodata, encode=encode_nodata)`.
        exclude_value : Optional[Number], default = None
            Value to exclude from crosstab. This could be used to denote a no data value if masking wasn't used. By default, NaNs are not cross-tabulated.
        negative_categories: Union[Number, Iterable[Number], Dict[str, Union[Number, Iterable[Number]]]], default=None
            Number or list of numbers representing the values to consider as the negative condition. This should be set to None when no negative categories are used or when the average type is "macro" or "weighted".
        average : str, default = "micro"
            Type of average to use when computing metrics. Options are "micro", "macro", and "weighted".
            Micro weighing computes the conditions, tp, tn, fp, and fn, for each category and then sums them.
            Macro weighing computes the metrics for each category then averages them.
            Weighted average computes the metrics for each category then averages them weighted by the number of weights argument in each category.
        weights : Optional[Iterable[Number]], default = None
            Weights to use when computing weighted average, specifically when the average argument is "weighted". Elements correspond to positive categories in order.

            Example:

            `positive_categories = [1, 2]; weights = [0.25, 0.75]`
        rasterize_attributes: Optional[list], default = None
            Numerical attributes of a Benchmark Map GeoDataFrame to rasterize.  Only applicable if benchmark map is a vector file.
            This cannot be none if the benchmark map is a vector file.
        attribute_tracking: bool, default = False
            Whether to return a dataframe with the attributes of the candidate and benchmark maps.
        attribute_tracking_kwargs: Optional[Dict], default = None
            Keyword arguments to pass to `gval.attribute_tracking()`.  This is only used if `attribute_tracking` is True. By default, agreement maps are used for attribute tracking but this can be set to None within this argument to override. See `gval.attribute_tracking` for more information.
        subsampling_df: Optional[gpd.GeoDataFrame], default = None
            DataFrame with spatial geometries and method types to subsample
        subsampling_average: Optional[str], default = None
            Way to aggregate statistics for subsamples if provided. Options are "sample", "band", and "full-detail"
            Sample calculates metrics and averages the results by subsample
            Band calculates metrics and averages all the metrics by band
            Full-detail does not aggregation on subsample or band

        Returns
        -------
        Union[
            Tuple[Union[xr.Dataset, xr.DataArray], DataFrame[Crosstab_df], DataFrame[Metrics_df]],
            Tuple[Union[xr.Dataset, xr.DataArray], DataFrame[Crosstab_df], DataFrame[Metrics_df], DataFrame[AttributeTrackingDf]]
        ]
            Tuple with agreement map/s, cross-tabulation table, and metric table. Possibly attribute tracking table as well.
        """

        # using homogenize accessor to avoid code reuse
        candidate, benchmark = self._obj.gval.homogenize(
            benchmark_map, target_map, resampling, rasterize_attributes
        )

        agreement_map = candidate.gval.compute_agreement_map(
            benchmark_map=benchmark,
            comparison_function=comparison_function,
            pairing_dict=pairing_dict,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
            nodata=nodata,
            encode_nodata=encode_nodata,
            subsampling_df=subsampling_df,
        )

        crosstab_df = candidate.gval.compute_crosstab(
            agreement_map=agreement_map, subsampling_df=subsampling_df
        )

        metrics_df = _compute_categorical_metrics(
            crosstab_df=crosstab_df,
            metrics=metrics,
            positive_categories=positive_categories,
            negative_categories=negative_categories,
            average=average,
            weights=weights,
            sampling_average=subsampling_average,
        )

        vector_agreement = self.agreement_map_format == "vector"
        if attribute_tracking:
            results = self.__handle_attribute_tracking(
                candidate_map=candidate,
                benchmark_map=benchmark,
                agreement_map=agreement_map,
                attribute_tracking_kwargs=attribute_tracking_kwargs,
            )

            if len(results) == 2:
                attributes_df, agreement_map = results
            else:
                attributes_df = results

            agreement_map = (
                agreement_map.gval.vectorize_data()
                if vector_agreement
                else agreement_map
            )

            return agreement_map, crosstab_df, metrics_df, attributes_df

        agreement_map = (
            agreement_map.gval.vectorize_data() if vector_agreement else agreement_map
        )

        return agreement_map, crosstab_df, metrics_df

    def continuous_compare(
        self,
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray],
        metrics: Union[str, Iterable[str]] = "all",
        target_map: Optional[Union[xr.Dataset, str]] = "benchmark",
        resampling: Optional[Resampling] = Resampling.nearest,
        nodata: Optional[Number] = None,
        encode_nodata: Optional[bool] = False,
        rasterize_attributes: Optional[list] = None,
        attribute_tracking: bool = False,
        attribute_tracking_kwargs: Optional[Dict] = None,
        subsampling_df: Optional[gpd.GeoDataFrame] = None,
        subsampling_average: str = "none",
    ) -> Tuple[
        Union[
            Tuple[Union[xr.Dataset, xr.DataArray], DataFrame[Metrics_df]],
            Tuple[
                Union[xr.Dataset, xr.DataArray],
                DataFrame[Metrics_df],
                DataFrame[AttributeTrackingDf],
            ],
        ]
    ]:
        """
        Computes comparison between two continuous value xarray's.

        Conducts the following steps:
            - homogenize: aligns data types, spatial alignment, and rasterizes data
            - compute_agreement: computes agreement map which is error or candidate minus benchmark
            - compute_metrics: computes metrics

        Spatially aligning the xarray's produces copies of the original candidate and benchmark maps. To reduce memory usage, consider using the `homogenize()` accessor method to overwrite the original maps in memory or saving them on disk.

        Parameters
        ----------
        benchmark_map : Union[gpd.GeoDataFrame, xr.DataArray, xr.Dataset]
            Benchmark map.
        metrics: Union[str, Iterable[str]], default = "all"
            Statistics to return in metric table.  This can be seen with gval.ContStats.available_functions().
        target_map: Optional[Union[xr.Dataset, str]], default = "benchmark"
            xarray object to match the CRS's and coordinates of candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
        resampling : rasterio.enums.Resampling
            See :func:`rasterio.warp.reproject` for more details.
        nodata : Optional[Number], default = None
            No data value to write to agreement map output. This will use `rxr.rio.write_nodata(nodata)`.
        encode_nodata : Optional[bool], default = False
            Encoded no data value to write to agreement map output. A nodata argument must be passed. This will use `rxr.rio.write_nodata(nodata, encode=encode_nodata)`.
        rasterize_attributes: Optional[list], default = None
            Numerical attributes of a GeoDataFrame to rasterize.
        attribute_tracking: bool, default = False
            Whether to return a dataframe with the attributes of the candidate and benchmark maps.
        attribute_tracking_kwargs: Optional[Dict], default = None
            Keyword arguments to pass to `gval.attribute_tracking()`.  This is only used if `attribute_tracking` is True. By default, agreement maps are used for attribute tracking but this can be set to None within this argument to override. See `gval.attribute_tracking` for more information.
        subsampling_df: Optional[gpd.GeoDataFrame], default = None
            DataFrame with spatial geometries and method types to subsample
        subsampling_average: str, default = None
            Way to aggregate statistics for subsamples if provided. Options are "sample", "band", "weighted", and "none"
            Sample calculates metrics and averages the results by subsample
            Band calculates metrics and averages all the metrics by band
            Weighted calculates metrics, scales by the weight and then averages them based on the weights
            Full-detail provides full detailed table

        Returns
        -------
        Union[
            Tuple[Union[xr.Dataset, xr.DataArray], DataFrame[Metrics_df]],
            Tuple[Union[xr.Dataset, xr.DataArray], DataFrame[Metrics_df], DataFrame[AttributeTrackingDf]]
        ]
            Tuple with agreement map and metric table, possibly attribute tracking table as well.
        """

        # using homogenize accessor to avoid code reuse
        candidate, benchmark = self._obj.gval.homogenize(
            benchmark_map, target_map, resampling, rasterize_attributes
        )

        results = candidate.gval.compute_agreement_map(
            benchmark_map=benchmark,
            comparison_function=difference,
            nodata=nodata,
            encode_nodata=encode_nodata,
            subsampling_df=subsampling_df,
            continuous=True,
        )

        # If sampling_df return type gives three values assign all vars results, otherwise only agreement map results
        agreement_map, candidate_map, benchmark_map = (
            results if subsampling_df is not None else (results, candidate, benchmark)
        )

        metrics_df = _compute_continuous_metrics(
            agreement_map=agreement_map,
            candidate_map=candidate_map,
            benchmark_map=benchmark_map,
            metrics=metrics,
            subsampling_average=subsampling_average,
            subsampling_df=subsampling_df,
        )

        if attribute_tracking:
            results = self.__handle_attribute_tracking(
                candidate_map=candidate,
                benchmark_map=benchmark,
                agreement_map=agreement_map,
                attribute_tracking_kwargs=attribute_tracking_kwargs,
            )

            if len(results) == 2:
                attributes_df, agreement_map = results
            else:
                attributes_df = results

            return agreement_map, metrics_df, attributes_df

        return agreement_map, metrics_df

    def probabilistic_compare(
        self,
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray],
        metric_kwargs: dict,
        return_on_error: Optional[Any] = None,
        target_map: Optional[Union[xr.Dataset, str]] = "benchmark",
        resampling: Optional[Resampling] = Resampling.nearest,
        rasterize_attributes: Optional[list] = None,
        attribute_tracking: bool = False,
        attribute_tracking_kwargs: Optional[Dict] = None,
    ) -> DataFrame[Prob_metrics_df]:
        """
        Computes probabilistic metrics from candidate and benchmark maps.

        Parameters
        ----------
        benchmark_map : xr.DataArray or xr.Dataset
            Benchmark map.
        metric_kwargs : dict
            Dictionary of keyword arguments to metric functions. Keys must be metrics. Values are keyword arguments to metric functions. Don't pass keys or values for 'observations' or 'forecasts' as these are handled internally with `benchmark_map` and `candidate_map`, respectively. Available keyword arguments by metric are available in `DEFAULT_METRIC_KWARGS`. If values are None or empty dictionary, default values in `DEFAULT_METRIC_KWARGS` are used.
        return_on_error : Optional[Any], default = None
            Value to return within metrics dataframe if an error occurs when computing a metric. If None, the metric is not computed and None is returned. If 'error', the raised error is returned.
        target_map: Optional[xr.Dataset or str], default = "benchmark"
            xarray object to match the CRS's and coordinates of candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
        resampling : rasterio.enums.Resampling
            See :func:`rasterio.warp.reproject` for more details.
        nodata : Optional[Number], default = None
            No data value to write to agreement map output. This will use `rxr.rio.write_nodata(nodata)`.
        encode_nodata : Optional[bool], default = False
            Encoded no data value to write to agreement map output. A nodata argument must be passed. This will use `rxr.rio.write_nodata(nodata, encode=encode_nodata)`.
        rasterize_attributes: Optional[list], default = None
            Numerical attributes of a GeoDataFrame to rasterize.
        attribute_tracking: bool, default = False
            Whether to return a dataframe with the attributes of the candidate and benchmark maps.
        attribute_tracking_kwargs: Optional[Dict], default = None
            Keyword arguments to pass to `gval.attribute_tracking()`.  This is only used if `attribute_tracking` is True. By default, agreement maps are used for attribute tracking but this can be set to None within this argument to override. See `gval.attribute_tracking` for more information.

        Returns
        -------
        DataFrame[Prob_metrics_df]
            Probabilistic metrics Pandas DataFrame with computed xarray's per metric and sample.

        Raises
        ------
        ValueError
            If keyword argument is required for metric but not passed.
            If keyword argument is not available for metric but passed.
            If metric is not available.

        Warns
        -----
        UserWarning
            Warns if a metric cannot be computed. `return_on_error` determines whether the metric is not computed and None is returned or if the raised error is returned.

        References
        ----------
        .. [1] `Scoring rule <https://en.wikipedia.org/wiki/Scoring_rule#Examples_of_proper_scoring_rules>`_
        .. [2] `Strictly Proper Scoring Rules, Prediction, and Estimation <https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf>`_
        .. [3] `Properscoring/properscoring <https://github.com/properscoring/properscoring>`_
        .. [4] `xskillscore/xskillscore <https://xskillscore.readthedocs.io/en/stable/api.html#probabilistic-metrics>`_
        .. [5] `What is a proper scoring rule? <https://statisticaloddsandends.wordpress.com/2021/03/27/what-is-a-proper-scoring-rule/>`_
        """

        # agreement_map temporarily None
        agreement_map = None

        # using homogenize accessor to avoid code reuse
        candidate, benchmark = self._obj.gval.homogenize(
            benchmark_map, target_map, resampling, rasterize_attributes
        )

        metrics_df = _compute_probabilistic_metrics(
            candidate_map=candidate,
            benchmark_map=benchmark,
            metric_kwargs=metric_kwargs,
            return_on_error=return_on_error,
        )

        if attribute_tracking:
            results = self.__handle_attribute_tracking(
                candidate_map=candidate,
                benchmark_map=benchmark,
                agreement_map=agreement_map,
                attribute_tracking_kwargs=attribute_tracking_kwargs,
            )

            if len(results) == 2:
                # attributes_df, agreement_map = results
                attributes_df, _ = results
            else:
                attributes_df = results

            return agreement_map, metrics_df, attributes_df

        return agreement_map, metrics_df

    def homogenize(
        self,
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray],
        target_map: Optional[Union[xr.Dataset, str]] = "benchmark",
        resampling: Optional[Resampling] = Resampling.nearest,
        rasterize_attributes: Optional[list] = None,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Homogenize candidate and benchmark maps to prepare for comparison.

        Currently supported operations include:
            - Matching projections and coordinates (spatial alignment)
            - Homogenize file formats (xarray/rasters)
            - Homogenize numerical data types (int, float, etc.).

        Parameters
        ----------
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray]
            Benchmark map.
        target_map: Optional[Union[xr.DataArray, xr.Dataset, str]], default = "benchmark"
            xarray object to match candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
        resampling: rasterio.enums.Resampling
            See :func:`rasterio.warp.reproject` for more details.
        rasterize_attributes: Optional[list], default = None
            Numerical attributes of a GeoDataFrame to rasterize

        Returns
        --------
        Union[xr.Dataset, xr.DataArray]
            Tuple with candidate and benchmark map respectively.
        """

        if isinstance(benchmark_map, gpd.GeoDataFrame):
            benchmark_map = _rasterize_data(
                candidate_map=self._obj,
                benchmark_map=benchmark_map,
                rasterize_attributes=rasterize_attributes,
            )
            self.agreement_map_format = "vector"

        self.check_same_type(benchmark_map)

        candidate, benchmark = _align_numeric_data_type(
            candidate_map=self._obj, benchmark_map=benchmark_map
        )

        candidate, benchmark = _spatial_alignment(
            candidate_map=candidate,
            benchmark_map=benchmark,
            target_map=target_map,
            resampling=resampling,
        )

        # For future operations
        candidate.gval.agreement_map_format = self.agreement_map_format

        return candidate, benchmark

    def compute_agreement_map(
        self,
        benchmark_map: Union[xr.Dataset, xr.DataArray],
        comparison_function: Union[
            Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str
        ] = "szudzik",
        pairing_dict: Optional[Dict[Tuple[Number, Number], Number]] = None,
        allow_candidate_values: Optional[Iterable[Union[int, float]]] = None,
        allow_benchmark_values: Optional[Iterable[Union[int, float]]] = None,
        nodata: Optional[Number] = None,
        encode_nodata: Optional[bool] = False,
        subsampling_df: Optional[gpd.GeoDataFrame] = None,
        continuous: bool = False,
    ) -> Union[Union[xr.Dataset, xr.DataArray, List[Union[xr.Dataset, xr.DataArray]]]]:
        """
        Computes agreement map as xarray from candidate and benchmark xarray's.

        Parameters
        ----------
        benchmark_map : Union[xr.Dataset, xr.DataArray]
            Benchmark map.
        comparison_function : Union[Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str], default = 'szudzik'
            Comparison function. Created by decorating function with @nb.vectorize() or using np.ufunc(). Use of numba is preferred as it is faster. Strings with registered comparison_functions are also accepted. Possible options include "pairing_dict". If passing "pairing_dict" value, please see the description for the argument for more information on behaviour.
        pairing_dict: Optional[Dict[Tuple[Number, Number], Number]], default = None
            When "pairing_dict" is used for the comparison_function argument, a pairing dictionary can be passed by user. A pairing dictionary is structured as `{(c, b) : a}` where `(c, b)` is a tuple of the candidate and benchmark value pairing, respectively, and `a` is the value for the agreement array to be used for this pairing.

            If None is passed for pairing_dict, the allow_candidate_values and allow_benchmark_values arguments are required. For this case, the pairings in these two iterables will be paired in the order provided and an agreement value will be assigned to each pairing starting with 0 and ending with the number of possible pairings.

            A pairing dictionary can be used by the user to note which values to allow and which to ignore for comparisons. It can also be used to decide how nans are handled for cases where either the candidate and benchmark maps have nans or both.
        allow_candidate_values : Optional[Iterable[Union[int,float]]], default = None
            List of values in candidate to include in computation of agreement map. Remaining values are excluded. If "pairing_dict" is set selected for comparison_function and pairing_function is None, this argument is necessary to construct the dictionary. Otherwise, this argument is optional and by default this value is set to None and all values are considered.
        allow_benchmark_values : Optional[Iterable[Union[int,float]]], default = None
            List of values in benchmark to include in computation of agreement map. Remaining values are excluded. If "pairing_dict" is set selected for comparison_function and pairing_function is None, this argument is necessary to construct the dictionary. Otherwise, this argument is optional and by default this value is set to None and all values are considered.
        nodata : Optional[Number], default = None
            No data value to write to agreement map output. This will use `rxr.rio.write_nodata(nodata)`.
        encode_nodata : Optional[bool], default = False
            Encoded no data value to write to agreement map output. A nodata argument must be passed. This will use `rxr.rio.write_nodata(nodata, encode=encode_nodata)`.
        subsampling_df : Optional[gpd.GeoDataFrame], default = None
            DataFrame with geometries to subsample data with or use as an exclusionary mask
        continuous :  bool, default = False
            Whether to return modified candidate and benchmark maps

        Returns
        -------
        Union[Union[xr.Dataset, xr.DataArray, List[Union[xr.Dataset, xr.DataArray]]]]
            Agreement map.
        """

        self.check_same_type(benchmark_map)

        agreement_range = 1 if subsampling_df is None else subsampling_df.shape[0]

        agreement_maps, candidate_maps, benchmark_maps, sample_percentage = (
            [],
            [],
            [],
            [],
        )
        for idx in range(agreement_range):
            results = (
                subsample(
                    candidate=self._obj,
                    benchmark=benchmark_map,
                    subsampling_df=subsampling_df.iloc[idx : idx + 1, :],
                )
                if subsampling_df is not None
                else [[self._obj, benchmark_map]]
            )

            candidate_copy, benchmark_copy = results[0]
            if "sample_percentage" in candidate_copy.attrs:
                sample_percentage.append(candidate_copy.attrs["sample_percentage"])

            agreement_map = Comparison.process_agreement_map(
                candidate_map=candidate_copy,
                benchmark_map=benchmark_copy,
                comparison_function=comparison_function,
                pairing_dict=pairing_dict,
                allow_candidate_values=allow_candidate_values,
                allow_benchmark_values=allow_benchmark_values,
                nodata=nodata,
                encode_nodata=encode_nodata,
                continuous=continuous,
            )

            # Preserve sampled maps for continuous statistics, otherwise delete
            if continuous:
                candidate_maps.append(candidate_copy)
                benchmark_maps.append(benchmark_copy)
            else:
                del candidate_copy, benchmark_copy

            agreement_maps.append(agreement_map)

        if subsampling_df is not None:
            if "sample_percent" in subsampling_df.columns:  # pragma: no cover
                subsampling_df.drop(columns=["sample_percent"], inplace=True)
            subsampling_df.loc[:, "sample_percent"] = sample_percentage
            return (
                (agreement_maps, candidate_maps, benchmark_maps)
                if continuous
                else agreement_maps
            )

        return agreement_maps[0]

    def compute_crosstab(
        self,
        agreement_map: Optional[
            Union[xr.DataArray, xr.Dataset, Iterable[Union[xr.DataArray, xr.Dataset]]]
        ] = None,
        subsampling_df: Optional[gpd.GeoDataFrame] = None,
    ) -> DataFrame[Crosstab_df]:
        """
        Crosstab 2 or 3-dimensional xarray DataArray to produce Crosstab DataFrame.

        Parameters
        ----------
        agreement_map : Union[xr.Dataset, xr.DataArray], default = None
            Benchmark map, {dimension}-dimensional.

        subsampling_df: Optional[gpd.GeoDataFrame], default = None
            DataFrame with spatial geometries and method types to subsample


        Returns
        -------
        DataFrame[Crosstab_df]
            Crosstab DataFrame
        """

        # Use self if agreement_map argument is not provided otherwise use agreement_map parameter
        if agreement_map is not None:
            agreement_map = (
                agreement_map if isinstance(agreement_map, list) else [agreement_map]
            )
        else:
            agreement_map = [self._obj]

        # Create cross-tabulation table for each agreement map and concatenate them
        crosstabs = []
        for idx, agreement in enumerate(agreement_map):
            crosstab = (
                _crosstab_Datasets(agreement)
                if isinstance(self._obj, xr.Dataset)
                else _crosstab_DataArrays(agreement)
            )

            if subsampling_df is not None:
                crosstab.insert(
                    0, "subsample", subsampling_df.iloc[idx]["subsample_id"]
                )

            crosstabs.append(crosstab)

        return pd.concat(crosstabs)

    def attribute_tracking(
        self,
        benchmark_map: Union[xr.DataArray, xr.Dataset],
        agreement_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
        candidate_suffix: Optional[str] = "_candidate",
        benchmark_suffix: Optional[str] = "_benchmark",
        candidate_include: Optional[Iterable[str]] = None,
        candidate_exclude: Optional[Iterable[str]] = None,
        benchmark_include: Optional[Iterable[str]] = None,
        benchmark_exclude: Optional[Iterable[str]] = None,
    ) -> Union[
        DataFrame[AttributeTrackingDf],
        Tuple[DataFrame[AttributeTrackingDf], Union[xr.DataArray, xr.Dataset]],
    ]:
        """
        Concatenate xarray attributes into a single pandas dataframe.

        Parameters
        ----------
        candidate_map : Union[xr.DataArray, xr.Dataset]
            Self. Candidate map xarray object.
        benchmark_map : Union[xr.DataArray, xr.Dataset]
            Benchmark map xarray object.
        candidate_suffix : Optional[str], default = '_candidate'
            Suffix to append to candidate map xarray attributes, by default '_candidate'.
        benchmark_suffix : Optional[str], default = '_benchmark'
            Suffix to append to benchmark map xarray attributes, by default '_benchmark'.
        candidate_include : Optional[Iterable[str]], default = None
            List of attributes to include from candidate map. candidate_include and candidate_exclude are mutually exclusive arguments.
        candidate_exclude : Optional[Iterable[str]], default = None
            List of attributes to exclude from candidate map. candidate_include and candidate_exclude are mutually exclusive arguments.
        benchmark_include : Optional[Iterable[str]], default = None
            List of attributes to include from benchmark map. benchmark_include and benchmark_exclude are mutually exclusive arguments.
        benchmark_exclude : Optional[Iterable[str]], default = None
            List of attributes to exclude from benchmark map. benchmark_include and benchmark_exclude are mutually exclusive arguments.

        Raises
        ------
        ValueError
            If candidate_include and candidate_exclude are both not None.
        ValueError
            If benchmark_include and benchmark_exclude are both not None.

        Returns
        -------
        Union[DataFrame[AttributeTrackingDf], Tuple[DataFrame[AttributeTrackingDf], Union[xr.DataArray, xr.Dataset]]]
            Pandas dataframe with concatenated attributes from candidate and benchmark maps. If agreement_map is not None, returns a tuple with the dataframe and the agreement map.
        """
        return _attribute_tracking_xarray(
            candidate_map=self._obj,
            benchmark_map=benchmark_map,
            agreement_map=agreement_map,
            candidate_suffix=candidate_suffix,
            benchmark_suffix=benchmark_suffix,
            candidate_include=candidate_include,
            candidate_exclude=candidate_exclude,
            benchmark_include=benchmark_include,
            benchmark_exclude=benchmark_exclude,
        )

    def cat_plot(
        self,
        title: str = "Categorical Map",
        colormap: str = "viridis",
        figsize: Tuple[int, int] = None,
        legend_labels: list = None,
        plot_bands: Union[str, list] = "all",
        colorbar_label: Union[str, list] = "",
        basemap: xyzservices.lib.TileProvider = cx.providers.OpenStreetMap.Mapnik,
    ):
        """
        Plots categorical Map for xarray object

        Parameters
        __________
        title : str
            Title of map, default = "Categorical Map"
        colormap : str, default = "viridis"
            Colormap of data
        figsize : tuple[int, int], default=None
            Size of the plot
        legend_labels : list, default = None
            Override labels in legend
        plot_bands: Union[str, list], default='all'
            What bands to plot
        color_bar_label : Union[str, list], default =""
            Label or labels for colorbar in the case of continuous plots
        basemap : Union[bool, xyzservices.lib.TileProvider], default = cx.providers.Stamen.Terrain
            Add basemap to the plot

        References
        ----------
        .. [1] `Matplotlib figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html>`_
        .. [2] `Matplotlib legend <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_
        """

        return _map_plot(
            self._obj,
            title=title,
            colormap=colormap,
            figsize=figsize,
            legend_labels=legend_labels,
            plot_type="categorical",
            plot_bands=plot_bands,
            basemap=basemap,
            colorbar_label=colorbar_label,
        )

    def cont_plot(
        self,
        title: str = "Continuous Map",
        colormap: str = "viridis",
        figsize: Tuple[int, int] = None,
        plot_bands: Union[str, list] = "all",
        colorbar_label: Union[str, list] = "",
        basemap: xyzservices.lib.TileProvider = cx.providers.OpenStreetMap.Mapnik,
    ):
        """
        Plots categorical Map for xarray object

        Parameters
        __________
        title : str
            Title of map, default = "Categorical Map"
        colormap : str, default = "viridis"
            Colormap of data
        figsize : tuple[int, int], default=None
            Size of the plot
        plot_bands: Union[str, list], default='all'
            What bands to plot
        colorbar_label : Union[str, list], default =""
            Label or labels for colorbar in the case of continuous plots
        basemap : Union[bool, xyzservices.lib.TileProvider], default = cx.providers.Stamen.Terrain
            Add basemap to the plot

        References
        ----------
        .. [1] `Matplotlib figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html>`_
        .. [2] `Matplotlib legend <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_
        """

        return _map_plot(
            self._obj,
            title=title,
            colormap=colormap,
            figsize=figsize,
            plot_type="continuous",
            plot_bands=plot_bands,
            basemap=basemap,
            colorbar_label=colorbar_label,
        )

    def vectorize_data(self) -> gpd.GeoDataFrame:
        """
        Vectorize an xarray DataArray or Dataset

        Returns
        -------
        gpd.GeoDataFrame
            Vectorized data
        """

        return _vectorize_data(self._obj)
