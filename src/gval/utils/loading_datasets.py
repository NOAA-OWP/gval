"""
Functions to load datasets
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Union, Optional, Tuple, Dict, Any
import os

import rioxarray as rxr
import xarray as xr
import rasterio


def load_raster_as_xarray(
    filename: Union[
        str, os.PathLike, rasterio.io.DatasetReader, rasterio.vrt.WarpedVRT
    ],
    parse_coordinates: Optional[bool] = None,
    chunks: Optional[Union[int, Tuple, Dict]] = None,
    cache: Optional[bool] = None,
    lock: Optional[Any] = None,
    masked: Optional[bool] = False,
    mask_and_scale: Optional[bool] = False,
    default_name: Optional[str] = None,
    band_as_variable: Optional[bool] = False,
    **open_kwargs,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Wraps around :obj:`rioxarray.open_rasterio` providing control over some arguments.

    Parameters
    ----------
    filename : Union[ str, os.PathLike, rasterio.io.DatasetReader, rasterio.vrt.WarpedVRT ]
        Path to the file to open. Or already open rasterio dataset
    parse_coordinates : Optional[bool], default = None
        Whether to parse the x and y coordinates out of the file's
        ``transform`` attribute or not. The default is to automatically
        parse the coordinates only if they are rectilinear (1D).
        It can be useful to set ``parse_coordinates=False``
        if your files are very large or if you don't need the coordinates.
    chunks : Optional[Union[int, Tuple, Dict]], default = None
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``. If chunks is provided, it used to load the new
        DataArray into a dask array. Chunks can also be set to
        ``True`` or ``"auto"`` to choose sensible chunk sizes according to
        ``dask.config.get("array.chunk-size")``.
    cache : Optional[bool], default = None
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False.
    lock : Optional[Any], default = None
        If chunks is provided, this argument is used to ensure that only one
        thread per process is reading from a rasterio file object at a time.

        By default and when a lock instance is provided,
        a :class:`xarray.backends.CachingFileManager` is used to cache File objects.
        Since rasterio also caches some data, this will make repeated reads from the
        same object fast.

        When ``lock=False``, no lock is used, allowing for completely parallel reads
        from multiple threads or processes. However, a new file handle is opened on
        each request.
    masked: Optional[bool], default = False
        If True, read the mask and set values to NaN. Defaults to False.
    mask_and_scale: Optional[bool], default = False
        Lazily scale (using the `scales` and `offsets` from rasterio) and mask.
        If the _Unsigned attribute is present treat integer arrays as unsigned.
    default_name : Optional[str], default = None
        The name of the data array if none exists. Default is None.
    band_as_variable : Optional[bool], default = False
        If True, will load bands in a raster to separate variables.
    open_kwargs : kwargs, default = None
        Optional keyword arguments to pass into :func:`rasterio.open`.


    Returns
    -------
    Union[:obj:`xr.DataArray`, :obj:`xr.Dataset`]
        Loaded data.

    References
    ----------
    .. [1] [Rioxarray `open_rasterio`](https://corteva.github.io/rioxarray/stable/rioxarray.html)
    .. [2] [`rasterio.open()`](https://rasterio.readthedocs.io/en/stable/api/rasterio.html#rasterio.open)
    """

    return rxr.open_rasterio(
        filename=filename,
        parse_coordinates=parse_coordinates,
        cache=cache,
        lock=lock,
        default_name=default_name,
        band_as_variable=band_as_variable,
        masked=masked,
        mask_and_scale=mask_and_scale,
        **open_kwargs,
    )
