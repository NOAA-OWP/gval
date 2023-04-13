from typing import Iterable, Optional, Tuple, Union, Callable
from numbers import Number

import numpy as np
import numba as nb
import xarray as xr
from rasterio.enums import Resampling
from pandera.typing import DataFrame
import geopandas as gpd

from gval.homogenize.spatial_alignment import _spatial_alignment
from gval.homogenize.rasterize import _rasterize_data
from gval import Comparison
from gval.comparison.tabulation import _crosstab_Datasets, _crosstab_DataArrays
from gval.comparison.compute_categorical_metrics import _compute_categorical_metrics
from gval.utils.schemas import Crosstab_df, Metrics_df


class GVALXarray:
    """
    Class for extending xarray functionality

    Attributes
    ----------
    _obj : Union[xr.Dataset, xr.DataArray]
       Object to use off the accessor
    data_type : type
       Data type of the _obj
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.data_type = type(xarray_obj)

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

    def categorical_compare(
        self,
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray],
        comparison_function: Union[
            Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str
        ] = "szudzik",
        metrics: Union[str, Iterable[str]] = "all",
        target_map: Optional[Union[xr.Dataset, str]] = "benchmark",
        resampling: Optional[Resampling] = Resampling.nearest,
        pairing_dict: Optional[dict[Tuple[Number, Number], Number]] = None,
        allow_candidate_values: Optional[Iterable[Union[int, float]]] = None,
        allow_benchmark_values: Optional[Iterable[Union[int, float]]] = None,
        nodata: Optional[Number] = None,
        encode_nodata: Optional[bool] = False,
        exclude_value: Optional[Number] = None,
        positive_categories: Optional[Union[Number, Iterable[Number]]] = None,
        negative_categories: Optional[Union[Number, Iterable[Number]]] = None,
        rasterize_attributes: Optional[list] = None,
    ) -> Tuple[
        Union[xr.Dataset, xr.DataArray], DataFrame[Crosstab_df], DataFrame[Metrics_df]
    ]:
        """
        Processes alignment operation, computing agreement map, and cross tab dataframe


        Parameters
        ----------
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray]
            Benchmark map in xarray DataSet format.
        comparison_function : Union[Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str], default = 'szudzik'
            Comparison function. Created by decorating function with @nb.vectorize() or using np.ufunc(). Use of numba is preferred as it is faster. Strings with registered comparison_functions are also accepted. Possible options include "pairing_dict". If passing "pairing_dict" value, please see the description for the argument for more information on behaviour.
        metrics: Union[str, Iterable[str]], default = "all"
            Statistics to return in metric table
        target_map: Optional[Union[xr.Dataset, str]], default = "benchmark"
            xarray object to match candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
        resampling : rasterio.enums.Resampling
            See :func:`rasterio.warp.reproject` for more details.
        pairing_dict: Optional[dict[Tuple[Number, Number], Number]], default = None
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
        exclude_value : Optional[Number], default = None
            Value to exclude from crosstab. This could be used to denote a no data value if masking wasn't used. By default, NaNs are not cross-tabulated.
        positive_categories: Optional[Union[Number, Iterable[Number]]], default = None
            Categories to represent positive entries
        negative_categories: Optional[Union[Number, Iterable[Number]]], default = None
            Categories to represent negative entries
        rasterize_attributes: Optional[list], default = None
            Attributes to rasterize from vector dataset

        Returns
        -------
        Union[xr.Dataset, xr.DataArray], DataFrame[Crosstab_df], DataFrame[Metrics_df]
            Tuple with agreement map, cross-tabulation table, and metric table
        """
        benchmark_map = _rasterize_data(
            candidate_map=self._obj,
            benchmark_map=benchmark_map,
            rasterize_attributes=rasterize_attributes,
        )

        self.check_same_type(benchmark_map)

        candidate, benchmark = _spatial_alignment(
            candidate_map=self._obj,
            benchmark_map=benchmark_map,
            target_map=target_map,
            resampling=resampling,
        )

        agreement_map = Comparison.process_agreement_map(
            candidate_map=candidate,
            benchmark_map=benchmark,
            comparison_function=comparison_function,
            pairing_dict=pairing_dict,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
            nodata=nodata,
            encode_nodata=encode_nodata,
        )

        if isinstance(self._obj, xr.Dataset):
            crosstab_df = _crosstab_Datasets(
                candidate_map=candidate,
                benchmark_map=benchmark_map,
                allow_candidate_values=allow_candidate_values,
                allow_benchmark_values=allow_benchmark_values,
                exclude_value=exclude_value,
            )

        else:
            crosstab_df = _crosstab_DataArrays(
                candidate_map=candidate,
                benchmark_map=benchmark_map,
                allow_candidate_values=allow_candidate_values,
                allow_benchmark_values=allow_benchmark_values,
                exclude_value=exclude_value,
            )

        metrics_df = _compute_categorical_metrics(
            crosstab_df=crosstab_df,
            metrics=metrics,
            positive_categories=positive_categories,
            negative_categories=negative_categories,
        )

        return agreement_map, crosstab_df, metrics_df

    def spatial_alignment(
        self,
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray],
        target_map: Optional[Union[xr.Dataset, str]] = "benchmark",
        resampling: Optional[Resampling] = Resampling.nearest,
        rasterize_attributes: Optional[list] = None,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Reproject :class:`xarray.Dataset` objects

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Others are appended as is.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatialreference.org), make sure it is an OGC WKT.

        Parameters
        ----------
        benchmark_map: Union[gpd.GeoDataFrame, xr.Dataset, xr.DataArray]
            Benchmark map in xarray DataArray format.
        target_map: Optional[Union[xr.DataArray, xr.Dataset, str]], default = "benchmark"
            xarray object to match candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
        resampling: rasterio.enums.Resampling
            See :func:`rasterio.warp.reproject` for more details.
        rasterize_attributes: Optional[list], default = None
            Attributes to rasterize from vector dataset


        Returns
        --------
        Union[xr.Dataset, xr.DataArray]
            Tuple with candidate and benchmark map respectively.
        """
        benchmark_map = _rasterize_data(
            candidate_map=self._obj,
            benchmark_map=benchmark_map,
            rasterize_attributes=rasterize_attributes,
        )

        self.check_same_type(benchmark_map)

        candidate, benchmark = _spatial_alignment(
            candidate_map=self._obj,
            benchmark_map=benchmark_map,
            target_map=target_map,
            resampling=resampling,
        )

        return candidate, benchmark

    def compute_agreement_map(
        self,
        benchmark_map: xr.Dataset,
        comparison_function: Union[
            Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str
        ] = "szudzik",
        pairing_dict: Optional[dict[Tuple[Number, Number], Number]] = None,
        allow_candidate_values: Optional[Iterable[Union[int, float]]] = None,
        allow_benchmark_values: Optional[Iterable[Union[int, float]]] = None,
        nodata: Optional[Number] = None,
        encode_nodata: Optional[bool] = False,
    ) -> xr.Dataset:
        """
        Computes agreement map as xarray from candidate and benchmark xarray's.

        Parameters
        ----------
        benchmark_map : Union[xr.Dataset, xr.DataArray]
            Benchmark map.
        comparison_function : Union[Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str], default = 'szudzik'
            Comparison function. Created by decorating function with @nb.vectorize() or using np.ufunc(). Use of numba is preferred as it is faster. Strings with registered comparison_functions are also accepted. Possible options include "pairing_dict". If passing "pairing_dict" value, please see the description for the argument for more information on behaviour.
        pairing_dict: Optional[dict[Tuple[Number, Number], Number]], default = None
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

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            Agreement map.
        """
        self.check_same_type(benchmark_map)

        return Comparison.process_agreement_map(
            candidate_map=self._obj,
            benchmark_map=benchmark_map,
            comparison_function=comparison_function,
            pairing_dict=pairing_dict,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
            nodata=nodata,
            encode_nodata=encode_nodata,
        )

    def compute_crosstab(
        self,
        benchmark_map: Union[xr.Dataset, xr.DataArray],
        allow_candidate_values: Optional[Iterable[Number]] = None,
        allow_benchmark_values: Optional[Iterable[Number]] = None,
        exclude_value: Optional[Number] = None,
    ) -> DataFrame:
        """
        Crosstab 2 or 3-dimensional xarray DataArray to produce Crosstab DataFrame.

        Parameters
        ----------
        benchmark_map : Union[xr.Dataset, xr.DataArray]
            Benchmark map, {dimension}-dimensional.
        allow_candidate_values : Optional[Iterable[Union[int,float]]], default = None
            Sequence of values in candidate to include in crosstab. Remaining values are excluded.
        allow_benchmark_values : Optional[Iterable[Union[int,float]]], default = None
            Sequence of values in benchmark to include in crosstab. Remaining values are excluded.
        exclude_value : Optional[Number], default = None
            Value to exclude from crosstab. This could be used to denote a no data value if masking wasn't used. By default, NaNs are not cross-tabulated.

        Returns
        -------
        Crosstab DataFrame
        """
        self.check_same_type(benchmark_map)

        if isinstance(self._obj, xr.Dataset):
            return _crosstab_Datasets(
                self._obj,
                benchmark_map,
                allow_candidate_values,
                allow_benchmark_values,
                exclude_value,
            )
        else:
            return _crosstab_DataArrays(
                self._obj,
                benchmark_map,
                allow_candidate_values,
                allow_benchmark_values,
                exclude_value,
            )
