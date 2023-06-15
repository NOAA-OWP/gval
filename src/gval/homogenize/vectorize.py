from typing import Union

import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from rasterio.features import shapes
import xarray as xr


def _vectorize_data(
    raster_data: Union[xr.Dataset, xr.DataArray],
) -> gpd.GeoDataFrame:
    """

    Parameters
    ----------
    raster_data: Union[xr.Dataset, xr.DataArray]
        Raster map to convert to vectorized map

    Returns
    -------
    gpd.GeoDataFrame
        Vectorized data

    """

    # Allowed numerical datatypes in vectorize operation
    dtypes = [np.int16, np.int32, np.uint8, np.uint16, np.float32]

    def get_dtype(dtype):
        """
        Conform datatype to alloable vectorize dtype

        Parameters
        ----------
        dtype : type
            Numerical datatype of raster data

        Returns
        -------
        type
            Type to use for vectorizing
        """

        if dtype not in dtypes:
            if np.issubdtype(dtype, np.integer):
                dtype = np.int32
            else:
                dtype = np.float32

        return dtype

    # Get bands, nodata, and dtype for data array or every var from a dataset
    if isinstance(raster_data, xr.DataArray):
        if len(raster_data.shape) > 2:
            iter_bands = [
                f"band_{coord.values}" for coord in raster_data.coords["band"]
            ]
            indices = range(raster_data.shape[0])
        else:
            iter_bands, indices = ["band_1"], [slice(0, raster_data.shape[0])]
        nodata_vals = [raster_data.rio.nodata] * len(iter_bands)
        dtypes = [get_dtype(raster_data.dtype)] * len(iter_bands)

    else:
        iter_bands = indices = [d_var for d_var in raster_data.data_vars]
        nodata_vals = [raster_data[d_var].rio.nodata for d_var in raster_data.data_vars]
        dtypes = [
            get_dtype(raster_data[d_var].dtype) for d_var in raster_data.data_vars
        ]

    bands, values, geometry = [], [], []

    # Iterate over values and vectorize shapes
    for idx, band, nodata, dtype in zip(indices, iter_bands, nodata_vals, dtypes):
        shps = shapes(
            raster_data[idx].astype(np.float32),
            transform=raster_data[idx].rio.transform(),
        )

        for shp in shps:
            if shp[1] != nodata and not np.isnan(shp[1]):
                values.append(shp[1])
                geometry.append(shape(shp[0]))
                bands.append(band.split("_")[-1])

    gdf_vec = gpd.GeoDataFrame({"band": bands, "values": values, "geometry": geometry})
    gdf_vec.crs = raster_data[idx].rio.crs

    del raster_data

    return gdf_vec
