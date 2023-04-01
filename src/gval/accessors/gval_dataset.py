from typing import Iterable, Optional, Tuple, Union, Callable
from numbers import Number

import numpy as np
import numba as nb
import xarray as xr
from rasterio.enums import Resampling
from pandera.typing import DataFrame

from gval.homogenize.spatial_alignment import _spatial_alignment
from gval import Comparison
from gval.comparison.tabulation import _crosstab_Datasets
from gval.comparison.compute_categorical_metrics import _compute_categorical_metrics
from gval.utils.exceptions import RasterMisalignment
from gval.utils.schemas import Crosstab_df, Metrics_df


@xr.register_dataset_accessor("gval")
class GVALDataset:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.aligned = False

    def categorical_compare(
        self,
        benchmark_map: Union[xr.DataArray, xr.Dataset],
        comparison_function: Union[
            Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str
        ] = "szudzik",
        metrics: Union[str, Iterable[str]] = "all",
        target_map: Optional[Union[xr.DataArray, xr.Dataset, str]] = "benchmark",
        resampling: Optional[Resampling] = Resampling.nearest,
        pairing_dict: Optional[dict[Tuple[Number, Number], Number]] = None,
        allow_candidate_values: Optional[Iterable[Union[int, float]]] = None,
        allow_benchmark_values: Optional[Iterable[Union[int, float]]] = None,
        nodata: Optional[Number] = None,
        encode_nodata: Optional[bool] = False,
        exclude_value: Optional[Number] = None,
        positive_categories: Optional[Union[Number, Iterable[Number]]] = None,
        negative_categories: Optional[Union[Number, Iterable[Number]]] = None,
    ) -> Tuple[
        Union[xr.DataArray, xr.Dataset], DataFrame[Crosstab_df], DataFrame[Metrics_df]
    ]:
        """
        Processes alignment operation, computing agreement map, and cross tab dataframe


        Parameters
        ----------
        benchmark_map: Union[xr.DataArray, xr.Dataset]
            Benchmark map in xarray DataArray format.
        comparison_function : Union[Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str], default = 'szudzik'
            Comparison function. Created by decorating function with @nb.vectorize() or using np.ufunc(). Use of numba is preferred as it is faster. Strings with registered comparison_functions are also accepted. Possible options include "pairing_dict". If passing "pairing_dict" value, please see the description for the argument for more information on behaviour.
        metrics: Union[str, Iterable[str]], default = "all"
            Statistics to return in metric table
        target_map: Optional[Union[xr.DataArray, xr.Dataset, str]], default = "benchmark"
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

        Returns
        -------
        Tuple[Union[xr.DataArray, xr.Dataset], DataFrame[Crosstab_df], DataFrame[Metrics_df]
            Tuple with agreement map, cross-tabulation table, and metric table
        """

        candidate, benchmark = _spatial_alignment(
            candidate_map=self._obj,
            benchmark_map=benchmark_map,
            target_map=target_map,
            resampling=resampling,
        )
        candidate.gval.aligned, benchmark.gval.aligned = True, True

        agreement_map = Comparison.process_agreement_map(
            candidate,
            benchmark,
            comparison_function,
            pairing_dict=pairing_dict,
            allow_candidate_values=allow_candidate_values,
            allow_benchmark_values=allow_benchmark_values,
            nodata=nodata,
            encode_nodata=encode_nodata,
        )

        crosstab_df = _crosstab_Datasets(
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
        benchmark_map: Union[xr.DataArray, xr.Dataset],
        target_map: Optional[Union[xr.DataArray, xr.Dataset, str]] = "benchmark",
        resampling: Optional[Resampling] = Resampling.nearest,
    ) -> Tuple[Union[xr.DataArray, xr.Dataset], Union[xr.DataArray, xr.Dataset]]:
        """
        Reproject :class:`xarray.Dataset` objects

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Others are appended as is.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatialreference.org), make sure it is an OGC WKT.

        Parameters
        ----------
        benchmark_map: Union[xr.DataArray, xr.Dataset]
            Benchmark map in xarray DataArray format.
        target_map: Optional[Union[xr.DataArray, xr.Dataset, str]], default = "benchmark"
            xarray object to match candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
        resampling : rasterio.enums.Resampling
            See :func:`rasterio.warp.reproject` for more details.

        Returns
        --------
        Tuple[Union[xr.DataArray, xr.Dataset], Union[xr.DataArray, xr.Dataset]]
            Tuple with candidate and benchmark map respectively.
        """

        candidate, benchmark = _spatial_alignment(
            candidate_map=self._obj,
            benchmark_map=benchmark_map,
            target_map=target_map,
            resampling=resampling,
        )
        candidate.gval.aligned, benchmark.gval.aligned = True, True

        return candidate, benchmark

    def compute_agreement_map(
        self,
        benchmark_map: xr.DataArray,
        comparison_function: Union[
            Callable, nb.np.ufunc.dufunc.DUFunc, np.ufunc, np.vectorize, str
        ] = "szudzik",
        pairing_dict: Optional[dict[Tuple[Number, Number], Number]] = None,
        allow_candidate_values: Optional[Iterable[Union[int, float]]] = None,
        allow_benchmark_values: Optional[Iterable[Union[int, float]]] = None,
        nodata: Optional[Number] = None,
        encode_nodata: Optional[bool] = False,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Computes agreement map as xarray from candidate and benchmark xarray's.

        Parameters
        ----------
        benchmark_map : Union[xr.DataArray, xr.Dataset]
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
        Union[xr.DataArray, xr.Dataset]
            Agreement map.
        """

        if self.aligned and benchmark_map.gval.aligned:
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
        else:
            raise RasterMisalignment

    def compute_crosstab(
        self,
        benchmark_map: xr.DataArray,
        allow_candidate_values: Optional[Iterable[Number]] = None,
        allow_benchmark_values: Optional[Iterable[Number]] = None,
        exclude_value: Optional[Number] = None,
    ) -> DataFrame:
        """
        Crosstab 2 or 3-dimensional xarray DataArray to produce Crosstab DataFrame.

        Parameters
        ----------
        benchmark_map : {xarray_obj}
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

        if self.aligned and benchmark_map.gval.aligned:
            return _crosstab_Datasets(
                self._obj,
                benchmark_map,
                allow_candidate_values,
                allow_benchmark_values,
                exclude_value,
            )

        else:
            raise RasterMisalignment

    @property
    def is_aligned(self):
        return self.aligned
