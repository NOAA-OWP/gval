import xarray
import rasterio
import numpy as np
from gval.utils.loading_datasets import load_raster_as_xarray

def Raster_comparison( candidate_map, benchmark_map,
                       comparison_type='two-class',
                       metrics='two-class',
                       verbose=False,
                       compute=False
                     ):
    """
    Computes agreement raster between categorical candidate and benchmark maps.
    
    - Reads input rasters in chucks (blocks/windows).
    - Applies function to chunk and returns appropriate values (primary metrics, diff, etc).
    - Aggregate function to combine window scale outputs.
    - Returns agreement map and metrics.

    Parameters
    ----------
    candidate_map : str, os.PathLike, rasterio.io.DatasetReader, rasterio.io.WarpedVRT, xarray.Dataset, xarray.DataArray
       Candidate map
    benchmark_map : str, os.PathLike, rasterio.io.DatasetReader, rasterio.io.WarpedVRT, xarray.Dataset, xarray.DataArray
       Benchmark map
    comparison_type : str 
       Variable type. Accepts two-class, multi-class, continuous. When two-class is elected, multiple positive conditions are grouped together as a single positive condition for evaluation.
    
    Returns
    -------

    Raises
    ------

    Notes
    -----
    
    References
    ----------

    Examples
    --------
    """

    # temporary variables declaring args and kwargs to pass to rioxarray.open_rasterio
    loading_args = []
    loading_kwargs = {'chunks':[1,1,'auto'],'lock': None}
    
    # load maps to xarray
    candidate_map_xr = load_raster_as_xarray(candidate_map, *loading_args, **loading_kwargs)
    benchmark_map_xr = load_raster_as_xarray(benchmark_map, *loading_args, **loading_kwargs)

    # pre-comparison prep 

    # comparison
    # this will currently load everything.
    # how to avoid loading if only writing to disk?
    # comparison = two_class_comparison_with_negatives(candidate_map_xr, benchmark_map_xr)
    
    # if compute:
    #     with TqdmCallback(desc=,quiet=(not verbose):
    #         comparison.compute()

    return(agreement_map)
