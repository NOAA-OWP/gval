# spatial_alignment.py
"""
Functions to check for and ensure spatial alignment of two xarray DataArrays

# To-Do
- make functions that check indices
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Optional, Tuple, Union

import xarray as xr
from shapely.geometry import box
from rasterio.enums import Resampling
from odc.geo.xr import ODCExtensionDa

from gval.utils.loading_datasets import _handle_xarray_memory
from gval.utils.exceptions import RasterMisalignment, RastersDontIntersect

ODCExtensionDa


def _matching_crs(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> bool:
    """
    Checks candidate and benchmark maps for matching crs's.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    bool
        Matching CRS's or not.
    """
    match = candidate_map.rio.crs == benchmark_map.rio.crs
    return match


def _matching_spatial_indices(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    raise_exception: bool = False,
) -> bool:
    """
    Checks for matching indices in candidate and benchmark maps.

    TODO: How does this do with 3d DataArrays and Datasets?

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    bool
        Whether or not maps are perfectly aligned.

    Raises
    ------
    RasterMisalignment
    """
    try:
        xr.align(candidate_map, benchmark_map, join="exact")
    except ValueError:
        aligned = False
    else:
        aligned = True

    # raises exception if not aligned and desired to raise
    if (not aligned) & (raise_exception):
        raise RasterMisalignment

    return aligned


def _rasters_intersect(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    raise_exception: bool = False,
) -> bool:
    """
    Predicates spatial intersection of two `xr.DataArrays` or `xr.Datasets`.

    CRS's don't have to be the same.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    bool
        Tests spatial intersection.

    Raises
    ------
    RasterDontIntersect
        Rasters do not intersect.
    """

    # already matching crs's
    if _matching_crs(candidate_map, benchmark_map):
        candidate_map_bounds = candidate_map.rio.bounds()
        benchmark_map_bounds = benchmark_map.rio.bounds()

    # else transform bounds
    else:
        # use benchmark crs just to test for intersection
        target_crs = benchmark_map.rio.crs

        candidate_map_bounds = candidate_map.rio.transform_bounds(target_crs)
        benchmark_map_bounds = benchmark_map.rio.transform_bounds(target_crs)

    # convert bounds to shapely boxes
    candidate_map_box = box(*candidate_map_bounds)
    benchmark_map_box = box(*benchmark_map_bounds)

    # intersection as bool
    rasters_intersect_bool = candidate_map_box.intersects(benchmark_map_box)

    if (not rasters_intersect_bool) & raise_exception:
        raise RastersDontIntersect
    else:
        return rasters_intersect_bool


def _reproject_map(
    original_map: Union[xr.DataArray, xr.Dataset],
    target_map: Union[xr.DataArray, xr.Dataset],
    resampling: str,
) -> Union[xr.DataArray, xr.Dataset]:
    """

    Parameters
    ----------
    original_map: Union[xr.DataArray, xr.Dataset]
        Map to be reprojected
    target_map: Union[xr.DataArray, xr.Dataset]
        Map to use for extent, resolution, and spatial reference
    resampling: str
        Method to resample changing resolutions

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Reprojected map
    """

    is_dst = isinstance(original_map, xr.Dataset)

    nodata = target_map["band_1"].rio.nodata if is_dst else target_map.rio.nodata
    reproj = original_map.odc.reproject(
        target_map.odc.geobox, tight=True, dst_nodata=nodata, resampling=resampling
    )

    # Coordinates need to be aligned
    reproj_coords = (
        reproj.rename({"longitude": "x", "latitude": "y"})
        if "longitude" in reproj.coords
        else reproj
    )

    del reproj

    #  Coordinates are virtually the same but 1e-8 or so is rounded differently
    final_reproj = reproj_coords.assign_coords(
        {"x": target_map.coords["x"], "y": target_map.coords["y"]}
    )
    del reproj_coords
    return final_reproj


def _align_rasters(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    target_map: Optional[Union[xr.DataArray, xr.Dataset, str]],
    resampling: Optional[Resampling] = Resampling.nearest,
) -> Tuple[Union[xr.DataArray, xr.Dataset], Union[xr.DataArray, xr.Dataset]]:
    """
    Reprojects raster to match a target map.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map. Nodata value should be set.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map. Nodata value should be set.
    target_map : Optional[Union[xr.DataArray, xr.Dataset, str]]
        Target map as string that can have values "candidate" or "benchmark". Otherwise pass an `xr.DataArray` or `xr.Dataset` to align maps to.
    resampling : rasterio.enums.Resampling
        See :func:`rasterio.warp.reproject` for more details.
    **kwargs : dict or keyword arguments
        Dictionary or keyword arguments to be passed `rio.xarray.reproject_match()`. DEPRECATED: This was found to conflict with rxr.rio.reproject_matchrk
        ().
    Returns
    -------
    Tuple[Union[xr.DataArray, xr.Dataset], Union[xr.DataArray, xr.Dataset]]
        Tuple with aligned candidate and benchmark map respectively.

    Raises
    ------
    ValueError
        target_map argument only accepts xr.DataArray, xr.Dataset, 'candidate', 'benchmark'.
    """

    def ensure_nodata_value_is_set(dataset_or_dataarray):
        """
        Ensure nodata values are set.
        NOTE: this prevents arbitrary large numbers from being introduced into the aligned map.
        """

        err_message = (
            "Both candidate and benchmark maps need to have nodata values set."
        )

        # dataarray case
        if isinstance(dataset_or_dataarray, xr.DataArray):
            if dataset_or_dataarray.rio.nodata is None:
                raise ValueError(err_message)

        # dataset case
        elif isinstance(dataset_or_dataarray, xr.Dataset):
            for var_name in dataset_or_dataarray.data_vars.keys():
                if dataset_or_dataarray[var_name].rio.nodata is None:
                    raise ValueError(err_message)

    # ensure both candidate and benchmarks have nodata values set
    ensure_nodata_value_is_set(candidate_map)
    ensure_nodata_value_is_set(benchmark_map)

    # already matching crs's and indices
    if _matching_crs(candidate_map, benchmark_map) & _matching_spatial_indices(
        candidate_map, benchmark_map
    ):
        return candidate_map, benchmark_map

    # align benchmark and candidate to target
    elif isinstance(target_map, (xr.DataArray, xr.Dataset)):
        candidate_map = _reproject_map(candidate_map, target_map, resampling)
        benchmark_map = _reproject_map(benchmark_map, target_map, resampling)

    # match candidate to benchmark
    elif target_map == "benchmark":
        candidate_map = _reproject_map(candidate_map, benchmark_map, resampling)

    # match benchmark to candidate
    elif target_map == "candidate":
        benchmark_map = _reproject_map(benchmark_map, candidate_map, resampling)

    else:
        raise ValueError(
            "Target_map argument only accepts xr.DataArray or xr.Dataset or a string with values of 'candidate' or 'benchmark'."
        )

    return candidate_map, benchmark_map


def _spatial_alignment(
    candidate_map: Union[xr.DataArray, xr.Dataset],
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
        If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

    Parameters
    ----------
    candidate_map: Union[xr.DataArray, xr.Dataset]
        Candidate map in xarray DataArray format.
    benchmark_map: Union[xr.DataArray, xr.Dataset]
        Benchmark map in xarray DataArray format.
    target_map: Optional[Union[xr.DataArray, xr.Dataset, str]], default = "benchmark"
        xarray object to match candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
    resampling : rasterio.enums.Resampling
        See :func:`rasterio.warp.reproject` for more details.
    **kwargs: dict, optional
        Additional keyword arguments to pass into :func:`rasterio.warp.reproject`.
        DEPRECATED: This was found to have issues with spatial alignment if certain arguments were passed.

    Returns
    --------
    Tuple[Union[xr.DataArray, xr.Dataset], Union[xr.DataArray, xr.Dataset]]
        Tuple with candidate and benchmark map respectively.

    Raises
    ------
    RasterDontIntersect
        Rasters do not intersect.

    References
    ----------
    .. [1] `xr.reproject_match <https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_dataset.RasterDataset.reproject>`_
    .. [2] `xr.reproject <https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_dataset.RasterDataset.reproject>`_
    .. [2] `rasterio.warp.reproject <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_

    """

    # check if rasters intersect at all
    _rasters_intersect(candidate_map, benchmark_map, raise_exception=True)

    # reproject maps to align
    cam, bem = _align_rasters(candidate_map, benchmark_map, target_map, resampling)

    return tuple(map(lambda x: _handle_xarray_memory(x, make_temp=True), [cam, bem]))
