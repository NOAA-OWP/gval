"""
Functions to load datasets
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Union, Optional, Tuple, Dict, Any
import os
import ast

import rioxarray as rxr
import xarray as xr
import rasterio
from tempfile import NamedTemporaryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


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

    .. deprecated:: 0.0.2
        `load_raster_as_xarray` will be removed in gval 0.0.2.  Use `rioxarray.open_rasterio` instead

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
    .. [1] `rioxarray.open_rasterio() <https://corteva.github.io/rioxarray/stable/rioxarray.html>`_
    .. [2] `rasterio.open() <https://rasterio.readthedocs.io/en/stable/api/rasterio.html#rasterio.open>`_
    """

    return rxr.open_rasterio(
        filename=filename,
        parse_coordinates=parse_coordinates,
        cache=cache,
        lock=lock,
        default_name=default_name,
        band_as_variable=band_as_variable,
        masked=masked,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **open_kwargs,
    )


_MEMORY_STRATEGY = "normal"


def adjust_memory_strategy(strategy: str):
    """
    Tells GVAL how to address handling memory.  There are three modes currently available:

    normal:  Keeps all of xarray files in memory as usual
    moderate:  Either creates cloud optimized geotiffs and stores as temporary files and reloads or reloads file
    to be in lazily loaded stated
    aggressive: Does the same as moderate except loads with no cache so everything is read from disk

    There are tradeoffs with performance for choosing a strategy that conserves memory, adjust only as needed.

    Parameters
    ----------
    strategy : str, {'normal', 'moderate', 'aggressive'}
        Method to conserve memory

    Raises
    ------
    ValueError

    """

    if strategy in {"normal", "moderate", "aggressive"}:
        global _MEMORY_STRATEGY
        _MEMORY_STRATEGY = strategy

    else:
        raise ValueError(
            "Please select one of the following options for a memory strategy "
            "'normal', 'moderate', 'aggressive'"
        )


def get_current_memory_strategy() -> str:
    """
    Gets the current memory_strategy

    Returns
    -------
    str
        Memory optimization strategy
    """
    return _MEMORY_STRATEGY


def _handle_xarray_memory(
    data_obj: Union[xr.Dataset, xr.DataArray], make_temp: bool = False
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Executes memory strategy plan depending on the set memory strategy.

    Nothing happens in this method when the plan is normal, only if the strategy is moderate or aggressive does
    any memory optimization occur.  If the parameter make_temp is true a temporary file with a lzw compressed
    cloud optimized geotiff will be made, otherwise the original file will be loaded.

    Parameters
    ----------
    data_obj : Union[xr.Dataset, xr.DataArray]
        Xarray object to handle_memory
    make_temp: bool
        Store data in a temporary file if in memory

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        Memory handled xarray object
    """

    # Check to see if it is already dask and what memory strategy is in place
    band_as_var = True if isinstance(data_obj, xr.Dataset) else False
    cache = True if _MEMORY_STRATEGY != "aggressive" else False

    if _MEMORY_STRATEGY == "normal" or _check_dask_array(data_obj):  # do nothing
        return data_obj
    elif (
        make_temp is False
    ):  # reload the file with cache on or off depending on strategy
        file_name = (
            data_obj["band_1"].encoding["source"]
            if band_as_var
            else data_obj.encoding["source"]
        )
        new_obj = _parse_string_attributes(
            rxr.open_rasterio(
                file_name,
                mask_and_scale=True,
                band_as_variable=band_as_var,
                cache=cache,
            )
        )
        del data_obj
        return new_obj

    else:  # make temporary file from lzw compressed COG and load with cache on or off depending on strategy
        dst_profile = cog_profiles.get("lzw")
        delete_file = not cache

        with NamedTemporaryFile(delete=True, suffix=".tif") as in_file:
            with NamedTemporaryFile(delete=delete_file, suffix=".tif") as out_file:
                data_obj.rio.to_raster(in_file.name, tiled=True, windowed=True)
                del data_obj
                cog_translate(in_file.name, out_file.name, dst_profile, in_memory=True)
                return _parse_string_attributes(
                    rxr.open_rasterio(
                        out_file.name,
                        mask_and_scale=True,
                        band_as_variable=band_as_var,
                        cache=cache,
                    )
                )


def _check_dask_array(original_map: Union[xr.DataArray, xr.Dataset]) -> bool:
    """
    Check whether map to be reprojected has dask data or not

    Parameters
    ----------
    original_map: Union[xr.DataArray, xr.Dataset]
        Map to be reprojected

    Returns
    -------
    bool
        Whether the data is a dask array
    """

    chunks = (
        original_map["band_1"].chunks
        if isinstance(original_map, xr.Dataset)
        else original_map.chunks
    )
    return chunks is not None


def _parse_string_attributes(
    obj: Union[xr.DataArray, xr.Dataset]
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Parses string attributes stored in rasters

    Parameters
    ----------
    obj: Union[xr.DataArray, xr.Dataset]
        Xarray object with possible string attributes

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Object returned with parsed attributes
    """

    if "pairing_dictionary" in obj.attrs and isinstance(
        obj.attrs["pairing_dictionary"], str
    ):
        eval_str = ast.literal_eval(
            obj.attrs["pairing_dictionary"].replace("nan", '"nan"')
        )
        obj.attrs["pairing_dictionary"] = {
            (float(k[0]), float(k[1])): float(v) for k, v in eval_str.items()
        }

        if isinstance(obj, xr.Dataset):
            for var in obj.data_vars:
                obj[var].attrs["pairing_dictionary"] = obj.attrs["pairing_dictionary"]

    if isinstance(obj, xr.Dataset):
        for var in obj.data_vars:
            if "pairing_dictionary" in obj[var].attrs and isinstance(
                obj[var].attrs["pairing_dictionary"], str
            ):
                eval_str = ast.literal_eval(
                    obj[var].attrs["pairing_dictionary"].replace("nan", '"nan"')
                )
                obj[var].attrs["pairing_dictionary"] = {
                    (float(k[0]), float(k[1])): float(v) for k, v in eval_str.items()
                }

    return obj
