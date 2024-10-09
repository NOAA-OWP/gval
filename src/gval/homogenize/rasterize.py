"""
Functions to rasterize vector datasets to process in raster space
"""

from numbers import Number
from typing import Union

import numpy as np
import xarray as xr
import geopandas as gpd
from geocube.api.core import make_geocube

from gval.utils.loading_datasets import _handle_xarray_memory


def _rasterize_data(
    candidate_map: Union[xr.Dataset, xr.DataArray],
    benchmark_map: gpd.GeoDataFrame,
    rasterize_attributes: list,
) -> Union[xr.Dataset, xr.DataArray]:
    """

    Parameters
    ----------
    candidate_map: Union[xr.Dataset, xr.DataArray]
        Candidate map to reference in creation of rasterized benchmark map
    benchmark_map: gpd.GeoDataFrame
        Benchmark map to be rasterized
    rasterize_attributes: list
        Attributes to rasterize

    Returns
    -------
    Union[xr.Dataset, xr.DataArray]
        Rasterized Data

    Raises
    ------
    KeyError

    """

    for attr in rasterize_attributes:
        if (
            not issubclass(type(benchmark_map[attr][0]), Number)
            or rasterize_attributes is None
            or len(rasterize_attributes) == 0
        ):
            raise KeyError(
                "Rasterize attribute needs to be not null and of numeric type"
            )

    rasterized_data = make_geocube(
        vector_data=benchmark_map,
        measurements=rasterize_attributes,
        like=candidate_map,
    )

    # Set nodata to value of candidate map
    if isinstance(candidate_map, xr.DataArray):
        rasterized_data = rasterized_data.to_array()
        # Nodata resolve
        rasterized_data.rio.write_nodata(np.nan, encoded=True, inplace=True)
        rasterized_data.rio.write_nodata(
            candidate_map.rio.encoded_nodata, encoded=True, inplace=True
        )

        if "band" in candidate_map.coords:
            # Deal with tabulation coords check (for single and multi-band examples)
            rasterized_data = rasterized_data.rename({"variable": "band"})
            rast_values = rasterized_data.values
            rasterized_data = rasterized_data.reindex(
                {"band": np.arange(rasterized_data.shape[0]) + 1}
            )
            rasterized_data.values = rast_values

        else:
            rasterized_data = rasterized_data.squeeze().drop_vars("variable")

    else:
        # make sure variable names are the same as would be loaded from rioxarray band_as_variable=True
        rasterized_data = rasterized_data.rename_vars(
            {
                k: v
                for k, v in zip(
                    rasterize_attributes,
                    [
                        f"band_{idx}"
                        for idx in np.arange(len(rasterized_data.data_vars)) + 1
                    ],
                )
            }
        )
        for var_name in rasterized_data.data_vars.keys():
            # Resolve nodata issues

            rasterized_data[var_name] = rasterized_data[var_name].rio.write_nodata(
                np.nan, encoded=True, inplace=True
            )

            if rasterized_data[var_name].rio.encoded_nodata is None:
                rasterized_data[var_name] = rasterized_data[var_name].rio.write_nodata(
                    candidate_map[var_name].rio.encoded_nodata,
                    encoded=True,
                    inplace=True,
                )

    return _handle_xarray_memory(rasterized_data, make_temp=True)
