"""
Functions to load datasets
"""

#__all__ = ['*']
__author__ = 'Fernando Aristizabal'

from typing import (
    Union
)
import os

import rioxarray as rioxarray
import xarray
import rasterio

# To-Do: allow for s3 reads
#import boto3


def load_raster_as_xarray(
    source: Union[str, os.PathLike, rasterio.io.DatasetReader,
             rasterio.vrt.WarpedVRT, xarray.DataArray],
    *args,
    **kwargs
    ) -> xarray.DataArray:
    """
    Loads a single raster as xarray DataArray from file path or URL.

    Currently working on extending support for S3.

    Parameters
    ----------
    source : str, os.PathLike, rasterio.io.DatasetReader, Rasterio.vrt.WarpedVRT, xarray.DataArray
        Path to file or opened Rasterio Dataset.
    *args : args, optional
        Optional positional arguments to pass to rioxarray.open_rasterio.
    *kwargs : kwargs, optional
        Optional keyword arguments to pass to rioxarray.open_rasterio.

    Returns
    -------
    xarray.DataArray
        xarray dataarray.
    """
    
    #if isinstance(source,(xarray.Dataset,xarray.DataArray)):
    
    # existing xarray DataArray
    if isinstance(source,xarray.DataArray):
        return source
    
    # local file path or S3 url
    elif isinstance(source,(str,os.PathLike)):
        # TO-DO: support authentication
        return rioxarray.open_rasterio(source, *args, **kwargs)
    
    # removed DataSet support for now
    # List[xarray.DataArray]
    #elif isinstance(source,list):
    #    if all( [isinstance(e,xarray.Dataset) for e in source] ):
    #        return source
    
    # if neither rasterio dataset, filepath, or url
    else:
        raise ValueError("Source should be a filepath to a raster or xarray Dataset, DataArray or list of Datasets.")

