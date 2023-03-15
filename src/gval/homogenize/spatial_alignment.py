# spatial_alignment.py
"""
Functions to check for and ensure spatial alignment of two xarray DataArrays

# To-Do
- make functions that check indices
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import (
    Optional,
    Tuple,
    Union,
)

import rasterio
import pyproj
import xarray as xr

from gval.utils.exceptions import RasterMisalignment


def matching_crs(
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


def matching_spatial_indices(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> bool:
    """
    Checks for matching indices in candidate and benchmark maps.

    TODO: How does this do with bands?

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
    """
    try:
        xr.align(candidate_map, benchmark_map, join="exact")
    except ValueError:
        return False
    else:
        return True


def transform_bounds(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    target_crs: Union[rasterio.crs.CRS, pyproj.crs.CRS, dict, str],
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """
    Transforms bounds of xarray datasets to target map or desired CRS.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.
    target_crs : Union[rasterio.crs.CRS, pyproj.crs.CRS, dict, str]
        Target CRS to project bounds to. Can be a string denoting 'candidate', 'benchmark', OGC WKT string, or Proj.4 string. Passing 'candidate' or 'benchmark' will transform both bounds to the CRS of that respective xarray.Additionally, values accepted by the argument `dst_crs` in :func:`rasterio.reproject` are allowed including :obj:`rasterio.crs.CRS`, :obj:`dict`, or :obj:`pyproj.crs.CRS`.

    Returns
    -------
    Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]
        Transformed bounds in tuple form for candidate and benchmark respectively.
    """

    # already matching crs's
    if matching_crs(candidate_map, benchmark_map):
        return (candidate_map.rio.bounds(), benchmark_map.rio.bounds())

    # match candidate to benchmark
    elif target_crs == "benchmark":
        return (
            candidate_map.rio.transform_bounds(benchmark_map.rio.crs),
            benchmark_map.rio.bounds(),
        )

    # match benchmark to candidate
    elif target_crs == "candidate":
        return (
            candidate_map.rio.bounds(),
            benchmark_map.rio.transform_bounds(candidate_map.rio.crs),
        )

    # match candidate to benchmark
    else:
        return (
            candidate_map.rio.transform_bounds(target_crs),
            benchmark_map.rio.transform_bounds(target_crs),
        )


def rasters_intersect(
    candidate_map_bounds: Tuple[float, float, float, float],
    benchmark_map_bounds: Tuple[float, float, float, float],
) -> bool:
    """
    Checks if two rasters intersect spatially at all given their bounds.

    CRS' are assumed to be equal.

    Parameters
    ----------
    candidate_map_bounds : Tuple[float, float, float, float]
        Bounds for candidate map.
    benchmark_map_bounds : Tuple[float, float, float, float]
        Bounds for benchmark map.

    Returns
    -------
    bool
        Tests spatial intersection.
    """

    # convert bounds to shapely boxes
    c_min_x, c_min_y, c_max_x, c_max_y = candidate_map_bounds
    b_min_x, b_min_y, b_max_x, b_max_y = benchmark_map_bounds

    # check for intersection
    if ((b_min_x <= c_min_x <= b_max_x) | (b_min_x <= c_max_x <= b_max_x)) & (
        (b_min_y <= c_min_y <= b_max_y) | (b_min_y <= c_max_y <= b_max_y)
    ):
        return True
    else:
        return False


def align_rasters(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    target_map: Optional[Union[xr.DataArray, xr.Dataset, str]] = None,
    **kwargs,
) -> Tuple[xr.DataArray, xr.Dataset]:
    """
    Reprojects raster to match target map and/or override values.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.
    target_map : Optional[Union[xr.DataArray, xr.Dataset, str]], default = None
        Target map.

    Returns
    -------
    Tuple[xr.DataArray, xr.Dataset]
        Tuple with aligned candidate and benchmark map respectively.

    Raises
    ------
    ValueError
        If target_map is None, must pass kwargs for rasterio.warp.reproject function
    ValueError
        target_map argument only accepts xr.DataArray, xr.Dataset, 'candidate', 'benchmark', or None type.
    """

    # already matching crs's and indices
    if matching_crs(candidate_map, benchmark_map) & matching_spatial_indices(
        candidate_map, benchmark_map
    ):
        return candidate_map, benchmark_map

    # align benchmark and candidate to target
    elif isinstance(target_map, (xr.DataArray, xr.Dataset)):
        candidate_map = candidate_map.rio.reproject_match(target_map, **kwargs)
        benchmark_map = benchmark_map.rio.reproject_match(target_map, **kwargs)

    # match candidate to benchmark
    elif target_map == "benchmark":
        candidate_map = candidate_map.rio.reproject_match(benchmark_map, **kwargs)

    # match benchmark to candidate
    elif target_map == "candidate":
        benchmark_map = benchmark_map.rio.reproject_match(candidate_map, **kwargs)

    # no target passed
    elif target_map is None:
        # override values to pass to rasterio.warp.reproject
        if kwargs:
            candidate_map = candidate_map.rio.reproject(**kwargs)
            benchmark_map = benchmark_map.rio.reproject_match(candidate_map)
        else:  # must pass either target_map or kwargs
            raise ValueError(
                "If target_map is none, must pass kwargs for rasterio.warp.reproject function"
            )
    else:
        raise ValueError(
            "Target_map argument only accepts xr.DataArray, xr.Dataset, 'candidate', 'benchmark', or None type."
        )
    return candidate_map, benchmark_map


def Spatial_alignment(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    target_map: Optional[Union[xr.DataArray, xr.Dataset, str]] = "benchmark",
    **kwargs,
) -> Union[xr.DataArray, xr.DataArray]:
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
    **kwargs: dict, optional
        Additional keyword arguments to pass into :func:`rasterio.warp.reproject`.
        To override:
        - src_transform: `rio.write_transform`
        - src_crs: `rio.write_crs`
        - src_nodata: `rio.write_nodata`

    Returns
    --------
    Tuple[Union[xr.DataArray, xr.Dataset]]
        Tuple with candidate and benchmark map respectively.

    References
    ----------
    .. [1] [`xr.reproject_match](https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_dataset.RasterDataset.reproject)
    .. [2] [`xr.reproject](https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_dataset.RasterDataset.reproject)
    .. [2] [`rasterio.warp.reproject](https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject)

    """

    # transform bounds
    if "dst_crs" in kwargs:
        cam, bem = transform_bounds(candidate_map, benchmark_map, kwargs["dst_crs"])
    else:
        target_crs = (
            target_map.rio.crs
            if isinstance(target_map, (xr.DataArray, xr.Dataset))
            else target_map
        )
        cam, bem = transform_bounds(candidate_map, benchmark_map, target_crs)

    # check if rasters intersect at all
    if not rasters_intersect(cam, bem):
        raise RasterMisalignment

    # reproject maps to align
    cam, bem = align_rasters(candidate_map, benchmark_map, target_map, **kwargs)

    return cam, bem
