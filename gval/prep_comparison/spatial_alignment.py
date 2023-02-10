# spatial_alignment.py
"""
Functions to check for and ensure spatial alignment of two xarray DataArrays

# To-Do
- make functions that check indices
"""

#__all__ = ['*']
__author__ = 'Fernando Aristizabal'

from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import sys
import os

import xarray 
import rioxarray as rxr
import dask
from rasterio.enums import Resampling

# temporary
sys.path.append(os.path.abspath('../..'))

from gval.utils.exceptions import RasterMisalignment, RastersNotSingleBand

def matching_crs(
    candidate_map: xarray.DataArray,
    benchmark_map: xarray.DataArray,
    ) -> bool:
    """ Checks candidate and benchmark maps for matching crs's """

    match = candidate_map.rio.crs == benchmark_map.rio.crs

    return(match)

def checks_for_single_band(
    candidate_map: xarray.DataArray,
    benchmark_map: xarray.DataArray
    ) -> bool:
    """ Ensures single band rasters """

    if (candidate_map.indexes['band'] == 1) | (benchmark_map.indexes['band'] == 1):
        return True
    else:
        return False
    
def matching_spatial_indices(
    candidate_map: xarray.DataArray,
    benchmark_map: xarray.DataArray
    ) -> bool:
    """ Checks for matching indices in candidate and benchmark maps """
    
    # NEED TO REDO
        # CHECK FOR DIMENSIONS
        # floating point error

    # check for length of spatial dimensions
    matching_x_length = len(candidate_map.indexes['x']) == len(benchmark_map.indexes['x'])
    matching_y_length = len(candidate_map.indexes['y']) == len(benchmark_map.indexes['y'])

    # return false if length of spatial indices don't match
    if (not matching_x_length) | (not matching_y_length):
        return False
    
    # check index values 
    match = candidate_map.indexes == benchmark_map.indexes

    return(match)

def transform_bounds(
    candidate_map: xarray.DataArray,
    benchmark_map: xarray.DataArray,
    target_map: Optional[Union[xarray.DataArray,str]] = None,
    dst_crs: Optional[str] = None
    ) -> Tuple[Tuple[float, float, float, float],Tuple[float, float, float, float]]:
    """ Transforms bounds of xarray datasets to target map or desired crs """

    # already matching crs's
    if matching_crs(candidate_map, benchmark_map): 
        return(candidate_map.rio.bounds(), benchmark_map.rio.bounds())
    
    # match candidate to benchmark
    elif isinstance(target_map,xarray.DataArray) & (dst_crs != None):
        return( 
          candidate_map.rio.transform_bounds(target_map.rio.crs),
          benchmark_map.rio.transform_bounds(target_map.rio.crs)
          )
    
    # match candidate to benchmark
    elif (target_map == 'benchmark') & (dst_crs != None):
        return( 
          candidate_map.rio.transform_bounds(benchmark_map.rio.crs),
          benchmark_map.rio.bounds()
          )
    
    # match benchmark to candidate
    elif (target_map == 'candidate') & (dst_crs != None):
        return(
          candidate_map.rio.bounds(),
          benchmark_map.rio.transform_bounds(candidate_map.rio.crs)
          )
    
    # dst_crs is set
    elif (target_map == None) & (dst_crs != None):
        return(
          candidate_map.rio.transform_bounds(dst_crs),
          benchmark_map.rio.transform_bounds(dst_crs)
          )
    
    # both arguments are None types
    elif (target_map == None) & (dst_crs == None):
        raise ValueError("The arguments target_map and dst_crs cannot both be None types.")
    
    # wrong argument value passed to target_map
    else:
        raise ValueError("target_map argument only accepts None type, 'candidate', or 'benchmark'.")
    
def rasters_intersect(
    candidate_map_bounds: Tuple[float, float, float, float],
    benchmark_map_bounds: Tuple[float, float, float, float]
    ) -> bool:

    """ Checks if two rasters intersect spatially at all given their bounds"""

    # convert bounds to shapely boxes
    c_min_x, c_min_y, c_max_x, c_max_y = candidate_map_bounds
    b_min_x, b_min_y, b_max_x, b_max_y = benchmark_map_bounds

    # check for intersection
    if (
         (b_min_x <= c_min_x <= b_max_x) |
         (b_min_x <= c_max_x <= b_max_x) |
         (b_min_y <= c_min_y <= b_max_y) |
         (b_min_y <= c_max_y <= b_max_y) 
       ):
        return True
    else:
        return False

def align_rasters(
    candidate_map: xarray.DataArray,
    benchmark_map: xarray.DataArray,
    target_map: Optional[Union[xarray.DataArray,str]] = None,
    **kwargs
    ) -> Tuple[xarray.DataArray,xarray.DataArray]:
    """ Reprojects raster to match target map and/or override values. """

    # already matching crs's and indices
    if matching_crs(candidate_map, benchmark_map) & matching_spatial_indices(candidate_map, benchmark_map): 
        return(candidate_map, benchmark_map)
    
    # match candidate to benchmark
    if target_map == 'benchmark':
        candidate_map = candidate_map.rio.reproject_match(benchmark_map,**kwargs)
    
    # match benchmark to candidate
    elif target_map == 'candidate':
        benchmark_map = benchmark_map.rio.reproject_match(candidate_map,**kwargs)
    
    # align benchmark and candidate to target
    elif isinstance(target_map,xarray.DataArray):
        candidate_map = candidate_map.rio.reproject_match(target_map,**kwargs)
        benchmark_map = benchmark_map.rio.reproject_match(target_map,**kwargs)
    
    # no target passed
    elif (target_map == None):
        # override values to pass to rasterio.warp.reproject
        if kwargs:
            candidate_map = candidate_map.rio.reproject(**kwargs)
            benchmark_map = benchmark_map.rio.reproject(**kwargs)
        else: # must pass either target_map or kwargs
            raise ValueError("If target_map is none, must pass kwargs for rasterio.warp.reproject function")
    else:
        raise ValueError("target_map argument only accepts xarray.DataArray, 'candidate', 'benchmark', or None type.")
    return(candidate_map, benchmark_map)

def Spatial_alignment( 
    candidate_map: xarray.DataArray,
    benchmark_map: xarray.DataArray,
    target_map: Optional[Union[xarray.DataArray,str]] = 'benchmark',
    **kwargs
    ) -> Union[xarray.DataArray,xarray.DataArray]:
    """
    Reproject :class:`xarray.Dataset` objects

    .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
        Others are appended as is.
        Requires either a grid mapping variable with 'spatial_ref' or
        a 'crs' attribute to be set containing a valid CRS.
        If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

    .. versionadded:: X.X.X.X feature

    Parameters
    ----------
    candidate_map: :class: `xarray.DataArray`
        Candidate map in xarray DataArray format.
    benchmark_map: :class: `xarray.DataArray`
        Benchmark map in xarray DataArray format.
    target_map: :class: `xarray.DataArray`, str, or None
        xarray.DataArray to match candidates and benchmarks to or str with 'candidate' or 'benchmark' as accepted values.
    **kwargs: dict
        Additional keyword arguments to pass into :func:`rasterio.warp.reproject`.
        To override:
        - src_transform: `rio.write_transform`
        - src_crs: `rio.write_crs`
        - src_nodata: `rio.write_nodata`

    Returns
    --------
    Tuple[:class:`xarray.DataArray`:,:class:`xarray.DataArray`:]
        Tuple with two xarray.DataArray elements.
    """
    
    # checks if rasters are single band
    if not checks_for_single_band(candidate_map,benchmark_map):
        raise RastersNotSingleBand

    # transform bounds
    if 'dst_crs' in kwargs:
        cam, bem = transform_bounds(candidate_map, benchmark_map, target_map, kwargs['dst_crs'])
    else:
        cam, bem = transform_bounds(candidate_map, benchmark_map, target_map)
    
    # check if rasters intersect at all
    if not rasters_intersect(cam, bem):
        raise RasterMisalignment

    # reproject maps to align
    cam, bem = align_rasters(candidate_map, benchmark_map, target_map, **kwargs)

    return cam, bem
