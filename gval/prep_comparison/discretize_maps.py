import xarray
import rasterio
import numpy as np
from gval.utils.loading_datasets import load_raster_as_xarray
from operator import eq, ne, gt, lt, le, ge
from gval.utils.misc_utils import isiterable

def Discretize_map( source_map, 
                    # support discretizing functions: equal, greater,less, greaterequal, lessequal,isclose
                    # accepts thresholds as single or series of values for each
                    # returns a private function with __equal(value_to_threshold)
                    # use operators above
                    operator,
                    positive_conditions=2,
                    negative_conditions=2,
                   ):

    """
    Discretize a map from continous value to multi- or two- class as well as multi- to two- class.
    
    Parameters
    ----------
    source_map : str, os.PathLike, rasterio.io.DatasetReader, rasterio.io.WarpedVRT, xarray.Dataset, xarray.DataArray
       Source map to discretize.
    positive_conditions : number or an iterable of numbers (float,int)
        Value or values representing positive condition in source map.
    candidate_conditions : number or an iterable of numbers (float,int)
        Value or values representing negative condition in candidate map.
    
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
    
    # check for iterables
    if isiterable(candidate_positive_conditions):
        pass
