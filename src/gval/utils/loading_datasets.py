"""Functions to load or create datasets"""
from __future__ import annotations

__author__ = "Fernando Aristizabal"


import warnings
from typing import Union, Optional, Tuple, Iterable
from numbers import Number
import ast

import rioxarray as rxr
import xarray as xr
import numpy as np
from tempfile import NamedTemporaryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import pystac_client

import stackstac

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
    original_map : Union[xr.DataArray, xr.Dataset]
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
    obj : Union[xr.DataArray, xr.Dataset]
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


def _convert_to_dataset(xr_object=Union[xr.DataArray, xr.Dataset]) -> xr.Dataset:
    """
    Converts xarray object to dataset if it is not one already.

    Parameters
    ----------
    xr_object : Union[xr.DataArray, xr.Dataset]
        Xarray object to convert or simply return

    Returns
    -------
    xr.Dataset
        Dataset object

    """

    if isinstance(xr_object, xr.DataArray):
        nodata = xr_object.rio.nodata

        xr_object = (
            xr_object.to_dataset(dim="band")
            if "band" in xr_object.dims
            else xr_object.to_dataset(name="1")
        )
        xr_object = xr_object.rename_vars({x: f"band_{x}" for x in xr_object.data_vars})

        # Account for nodata
        for var_name in xr_object.data_vars:
            xr_object[var_name] = xr_object[var_name].rio.write_nodata(nodata)

        return xr_object
    else:
        return xr_object


def _get_raster_band_nodata(band_metadata, nodata_fill) -> Number:
    """

    Parameters
    ----------
    band_metadata : list
        Metadata from raster:bands extension
    nodata_fill : Number
        Fill in value for missing data

    Returns
    -------
    Number
        Number representing nodata
    """

    if band_metadata:
        prop_string = str(band_metadata.coords["raster:bands"].values)
        idx1, idx2 = prop_string.find("{"), prop_string.rfind("}")

        return ast.literal_eval(prop_string[idx1 : idx2 + 1]).get("nodata")
    else:
        if nodata_fill is None:
            raise ValueError(
                "Must have nodata fill value if nodata is not present in metadata"
            )

        return nodata_fill


def _set_nodata(
    stack: xr.DataArray, band_metadata: list = None, nodata_fill: Number = None
) -> Number:
    """

    Parameters
    ----------
    stack : xr.DataArray
        Data to set nodata attribute
    band_metadata : list
        Metadata from raster:bands extension
    nodata_fill : Number
        Fill in value for missing data

    """

    if stack.rio.nodata is not None:
        stack.rio.write_nodata(stack.rio.nodata, inplace=True)
    elif stack.rio.encoded_nodata is not None:
        stack.rio.write_nodata(stack.rio.encoded_nodata, inplace=True)
    else:
        stack.rio.write_nodata(
            _get_raster_band_nodata(band_metadata, nodata_fill), inplace=True
        )


def _set_crs(stack: xr.DataArray, band_metadata: list = None) -> Number:
    """

    Parameters
    ----------
    stack : xr.DataArray
        Original data with no information
    band_metadata : dict
        Information with band metadata

    Returns
    -------
    Xarray DataArray with proper CRS

    """

    if stack.rio.crs is not None:
        return stack.rio.write_crs(stack.rio.crs)
    else:
        return stack.rio.write_crs(f"EPSG:{band_metadata['epsg'].values}")


def get_stac_data(
    url: str,
    collection: str,
    time: str,
    bands: list = None,
    query: str = None,
    time_aggregate: str = None,
    max_items: int = None,
    intersects: dict = None,
    bbox: list = None,
    resolution: int = None,
    nodata_fill: Number = None,
) -> xr.Dataset:
    """

    Parameters
    ----------
    url : str
        Address hosting the STAC API
    collection : str
        Name of collection to get (currently limited to one)
    time : str
        Single or range of values to query in the time dimension
    bands: list, default = None
        Bands to retrieve from service
    query : str, default = None
        String command to filter data
    time_aggregate : str, default = None
        Method to aggregate multiple time stamps
    max_items : int, default = None
        The maximum amount of records to retrieve
    intersects : dict, default = None
        Dictionary representing the type of geometry and its respective coordinates
    bbox : list, default = None
        Coordinates to filter the spatial range of request
    resolution : int, default = 10
        Resolution to get data from
    nodata_fill : Number, default = None
        Value to fill nodata where not present

    Returns
    -------
    xr.Dataset
        Xarray object with resepective STAC API data

    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Call cataloging url, search, and convert to xarray
        catalog = pystac_client.Client.open(url)

        stac_items = catalog.search(
            datetime=time,
            collections=[collection],
            max_items=max_items,
            intersects=intersects,
            bbox=bbox,
            query=query,
        ).get_all_items()

        stack = stackstac.stack(stac_items, resolution=resolution)

        # Only get unique time indices in case there are duplicates
        _, idxs = np.unique(stack.coords["time"], return_index=True)
        stack = stack[idxs]

        # Aggregate if there is more than one time
        if stack.coords["time"].shape[0] > 1:
            crs = stack.rio.crs
            if time_aggregate == "mean":
                stack = stack.mean(dim="time")
                stack.attrs["time_aggregate"] = "mean"
            elif time_aggregate == "min":
                stack = stack.min(dim="time")
                stack.attrs["time_aggregate"] = "min"
            elif time_aggregate == "max":
                stack = stack.max(dim="time")
                stack.attrs["time_aggregate"] = "max"
            else:
                raise ValueError("A valid aggregate must be used for time ranges")

            stack.rio.write_crs(crs, inplace=True)
        else:
            stack = stack[0]
            stack.attrs["time_aggregate"] = "none"

        # Select specific bands
        if bands is not None:
            bands = [bands] if isinstance(bands, str) else bands
            stack = stack.sel({"band": bands})

        band_metadata = (
            stack.coords["raster:bands"] if "raster:bands" in stack.coords else None
        )
        if "band" in stack.dims:
            og_names = [name for name in stack.coords["band"]]
            names = [f"band_{x + 1}" for x in range(len(stack.coords["band"]))]
            stack = stack.assign_coords({"band": names}).to_dataset(dim="band")

            for metadata, var, og_var in zip(band_metadata, stack.data_vars, og_names):
                _set_nodata(stack[var], metadata, nodata_fill)
                stack[var] = _set_crs(stack[var], band_metadata)
                stack[var].attrs["original_name"] = og_var

        else:
            stack = stack.to_dataset(name="band_1")
            _set_nodata(stack["band_1"], band_metadata, nodata_fill)
            stack["band_1"] = _set_crs(stack["band_1"])
            stack["band_1"].attrs["original_name"] = (
                bands[0] if isinstance(bands, list) else bands
            )

        return stack


def _create_circle_mask(
    sizes: int | Tuple[int], center: Tuple[Number, Number], radius: Number
) -> np.ndarray:
    """
    Function to create a circle mask

    Parameters
    ----------
    sizes : Int or Tuple of Int
        Size of xarray's in number of pixels or tuple with the x and y sizes, respectively.
    center : Tuple of Number
        Tuple with the center coordinates (x, y).
    radius : Number
        Radius of the circle.

    Returns
    -------
    mask : np.ndarray
        Numpy array with the circle mask.
    """
    Y, X = np.ogrid[: sizes[1], : sizes[0]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def _create_xarray(
    upper_left: Tuple[Number, Number],
    lower_right: Tuple[Number, Number],
    sizes: int | Tuple[int, int],
    band_params: Iterable[Tuple[Number, Number, Tuple[Number, Number], Number]],
    nodata_value: Optional[Number] = None,
    encoded_nodata_value: Optional[Number] = None,
    shapes: str = "circle",
    band_dim_name: str = "band",
    return_dataset: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Function to create xarray's with circles per band, and optionally return a Dataset

    Parameters
    ----------
    upper_left : Tuple of Number
        Tuple with the upper left coordinates (x, y).
    lower_right : Tuple of Number
        Tuple with the lower right coordinates (x, y).
    sizes : Int or Tuple of Int
        Size of xarray's in number of pixels or tuple with the x and y sizes, respectively.
    band_params : Iterable of Tuples
        Each tuple represents the parameters of each band comprised of:
        (background_value, circle_value, circle_center, circle_radius).

        Circle center is a tuple with (x, y) coordinates.

        The data types are:
            background_value: Number
            circle_value: Number
            circle_center: tuple of Number
            circle_radius: Number
    nodata_value : Optional[Number], default = None
        Nodata value to use for the xarray's.
    encoded_nodata_value : Optional[Number], default = None
        Nodata value to use for the encoded xarray's.
    shapes : str, default = 'circle'
        Shape of mask within the xarray's. Currently only 'circle' is supported.
    band_dim_name : str, default = 'band'
        Name of the band dimension.
    return_dataset : bool, default = False
        Whether to return a Dataset instead of a DataArray.

    Raises
    ------
    ValueError
        If the shape is not supported.

    Returns
    -------
    data_array : xarray.DataArray or xarray.Dataset
        Xarray with the circles per band.
    """

    # Handle sizes
    if isinstance(sizes, Number):
        sizes = int(sizes)
        sizes = (sizes, sizes)

    # handle shapes
    if shapes == "circle":
        mask_func = _create_circle_mask
    # elif False:
    # placeholder for future shapes
    # pass
    else:
        raise ValueError(f"Shape '{shapes}' is not supported.")

    # Create empty array
    array = np.full((len(band_params),) + tuple(reversed(sizes)), nodata_value)

    for band_idx, params in enumerate(band_params):
        background_value, circle_value, circle_center, circle_radius = params

        # Apply background value, leaving the border as nodata
        array[band_idx, 1:-1, 1:-1] = background_value

        if shapes == "circle":
            # Create circle mask and apply circle value
            circle_mask = mask_func(sizes, circle_center, circle_radius)

            array[band_idx, circle_mask] = circle_value

        # elif False:
        # placeholder for future shapes
        #    pass

    # create xarray DataArray
    data_array = xr.DataArray(data=array, dims=[band_dim_name, "y", "x"])
    data_array.rio.write_crs("epsg:4326", inplace=True)

    # assign nodata values
    if nodata_value is not None:
        data_array.rio.write_nodata(nodata_value, inplace=True)

    if encoded_nodata_value is not None:
        data_array.rio.write_nodata(encoded_nodata_value, inplace=True, encoded=True)

    # create coordinates
    lats = np.linspace(upper_left[1], lower_right[1], sizes[1])
    longs = np.linspace(lower_right[0], upper_left[0], sizes[0])
    bands = np.arange(1, len(band_params) + 1)

    # assign coordinates
    data_array = data_array.assign_coords({band_dim_name: bands, "y": lats, "x": longs})

    if return_dataset:
        # Convert the DataArray to a Dataset
        dataset = data_array.to_dataset(name="variable")
        return dataset

    return data_array


def _create_xarray_pairs(
    upper_left: Tuple[Number, Number],
    lower_right: Tuple[Number, Number],
    sizes: int | Tuple[int, int],
    band_params_candidate: Iterable[
        Tuple[Number, Number, Tuple[Number, Number], Number]
    ],
    band_params_benchmark: Iterable[
        Tuple[Number, Number, Tuple[Number, Number], Number]
    ],
    nodata_value: Optional[Number] = None,
    encoded_nodata_value: Optional[Number] = None,
    shapes: str = "circle",
    band_dim_name: str = "band",
    return_dataset: bool = False,
) -> Tuple[xr.DataArray | xr.Dataset]:
    """
    Function to create xarray's with shapes per band, and optionally return a Dataset

    Parameters
    ----------
    upper_left : Tuple of Number
        Tuple with the upper left coordinates (x, y).
    lower_right : Tuple of Number
        Tuple with the lower right coordinates (x, y).
    sizes : Int or Tuple of Int
        Size of xarray's in number of pixels or tuple with the x and y sizes, respectively.
    band_params : Iterable of Tuples
        Each tuple represents the parameters of each band comprised of:
        (background_value, circle_value, circle_center, circle_radius).

        Circle center is a tuple with (x, y) coordinates.

        The data types are:
            background_value: Number
            circle_value: Number
            circle_center: tuple of Number
            circle_radius: Number
    nodata_value : Optional[Number], default = None
        Nodata value to use for the xarray's.
    encoded_nodata_value : Optional[Number], default = None
        Nodata value to use for the encoded xarray's.
    shapes : str, default = 'circle'
        Shape of mask within the xarray's. Currently only 'circle' is supported.
    band_dim_name : str, default = 'band'
        Name of the band dimension.
    return_dataset : bool, default = False
        Whether to return a Dataset instead of a DataArray.

    Raises
    ------
    ValueError
        If the shape is not supported.

    Returns
    -------
    data_array : xarray.DataArray or xarray.Dataset
        Candidate map.
    data_array : xarray.DataArray or xarray.Dataset
        Benchmark map.
    """

    args = [
        upper_left,
        lower_right,
        sizes,
        None,
        nodata_value,
        encoded_nodata_value,
        shapes,
        band_dim_name,
        return_dataset,
    ]

    candidate_args = args.copy()
    benchmark_args = args.copy()

    candidate_args[3] = band_params_candidate
    benchmark_args[3] = band_params_benchmark

    candidate_map = _create_xarray(*candidate_args)
    benchmark_map = _create_xarray(*benchmark_args)

    return candidate_map, benchmark_map
