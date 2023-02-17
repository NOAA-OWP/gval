"""
Comparison functionality
 - Includes pairing and comparison functions
"""

#__all__ = ['*']
__author__ = 'Fernando Aristizabal'

from typing import (
    Any,
    Callable,
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
from numbers import Number

import numpy as np 
import pandas as pd
from xrspatial.zonal import crosstab
import xarray


def _are_not_natural_numbers(c : Number, b : Number) -> None:
    """ Checks pair to see if they are both natural numbers or two non-negative integers [0, 1, 2, 3, 4, ...) """
    for x in (c,b):
        if np.isnan(x): # checks to make sure it's not a nan value
            continue
        elif (x < 0) | ((x - int(x)) != 0): # checks for non-negative and whole number
            raise ValueError(f"{x} is not natural numbers (non-negative integers) [0, 1, 2, 3, 4, ...)")

## cantor method
def cantor_pair(c : int, b: int) -> int:
    """
    Produces unique natural number for two non-negative natural numbers (0,1,2,...) 

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.
    
    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    _are_not_natural_numbers(c, b)
    return .5 * (c**2 + c + 2*c*b + 3*b + b**2)

def szudzik_pair(c : int, b : int) -> int: 
    """
    Produces unique natural number for two non-negative natural numbers (0,1,2,3,...).

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.
    
    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    _are_not_natural_numbers(c, b)
    return c**2 + c + b if c >= b else b**2 + c

def negative_value_transformation(func : Callable):
    """
    Transforms negative values for use with pairing functions that only accept non-negative integers.

    Parameters
    ----------
    func : Callable
        Pairing function to apply negative value transformation to.

    Returns
    -------
    Callable
        Pairing function able to accept negative values.
    """
    _signing = lambda x : 2*x if x >= 0 else -2*x - 1
    
    def wrap(c,b):
        c = _signing(c)
        b = _signing(b)
        return func(c,b)

    return wrap

@negative_value_transformation
def cantor_pair_signed(c : int, b : int) -> int:
    """
    Output unique natural number for each unique combination of whole numbers using Cantor signed method.

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.
    
    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    return cantor_pair(c, b)

@negative_value_transformation
def szudzik_pair_signed(c : int, b : int) -> int:
    """
    Output unique natural number for each unique combination of whole numbers using Szudzik signed method._summary_

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.
    
    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    return szudzik_pair(c, b)

## user defined
def _make_pairing_dict(
    unique_candidate_values : Iterable, 
    unique_benchmark_values : Iterable
    ):
    """ Creates a dict pairing each unique value in candidate and benchmark arrays """
    from itertools import product
    pairing_dict = { k : v for v,(k,_) in enumerate(product(unique_candidate_values, unique_benchmark_values)) }

def pairing_dict_fn(c : int, b : int) -> int:
    """
    Produces a pairing dictionary that produces a unique result for every combination ranging from 0 to the number of combinations.

    Parameters
    ----------
    c : int
        Candidate map value.
    b : int
        Benchmark map value.

    Returns
    -------
    int
        Agreement map value.
    
    References
    ----------
    [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    return pairing_dict[c, b]

####################################
# compare numpy arrays

def compute_agreement_numpy(candidate : np.ndarray, benchmark : np.ndarray, 
                            comparison_function : callable
                           ) -> np.ndarray:
    """
    Computes agreement array from candidate and benchmark given a user provided comparison function.

    Parameters
    ----------
    candidate : np.ndarray
        Array of candidate map values.
    benchmark : np.ndarray
        Array of benchmark map values.
    comparison_function : callable
        Comparison function that inputs elements of candidate and benchmark and produces agreement map array values.

    Returns
    -------
    np.ndarray
        Agreement map array.
    """
    if not isinstance(comparison_function,np.ufunc):
        agreement =  np.frompyfunc(comparison_function,2,1)(candidate,benchmark)
    else:
        agreement = comparison_ufunc(candidate, benchmark)

    return agreement

def crosstab_xarray( candidate : xarray.DataArray,
                     benchmark : xarray.DataArray, 
                     agreement : xarray.DataArray,
                     allow_list_candidate : Optional[Iterable[Union[int,float]]] = None,
                     allow_list_benchmark : Optional[Iterable[Union[int,float]]] = None,
                     exclude_values : Optional[Union[int,float]] = None
                    ) -> pd.DataFrame:
    """
    Crosstab xarray DataArrays to produce crosstab df.

    Parameters
    ----------
    candidate : xarray.DataArray
        Candidate map
    benchmark : xarray.DataArray
        Benchmark map
    agreement : xarray.DataArray
        Agreement map
    allow_list_candidate : Optional[Iterable[Union[int,float]]], optional
        List of values in candidate to include in crosstab. Remaining values are excluded, by default None
    allow_list_benchmark : Optional[Iterable[Union[int,float]]], optional
        List of values in benchmark to include in crosstab. Remaining values are excluded, by default None
    exclude_values : Optional[Union[int,float]], optional
        List of values to exclude from crosstab, by default None

    Returns
    -------
    xarray.DataArray
        _description_
    """

    # according to xarray-spatial docs: 
    # Nodata value in values raster. Cells with nodata do not belong to any zone, and thus excluded from calculation.
    # would this be necessary? Does this just provide another means to exclude??
    crosstab_df = crosstab(zones = candidate, values = benchmark,
                           zone_ids = allow_list_candidate,
                           cat_ids = allow_list_benchmark,
                           nodata_values = exclude
    )
             

    


def crosstab_numpy(candidate : np.ndarray, benchmark : np.ndarray, agreement : np.ndarray,
                   comparison_function : callable, preserve_axis : Union[int,None] = None
                  ):

    """ Computes crosstab dataframe with candidate, benchmark and agreement arrays based on comparison function """

    # how to crosstab by axis???? np.apply_over_axis???

    # get unique values for three arrays
    unique_candidate = np.unique(candidate)
    unique_benchmark = np.unique(benchmark)

    # make a dict with {agreement values : (candidate, benchmark)} and it's inverse
    crosstab_key = { comparison_function(c, b) : (c, b) for c, b in zip(unique_candidate, unique_benchmark) }
    crosstab_key_flip = { v : k for k, v in crosstab_key.iteritems() }

    # get dict of unique agreement values as keys and their counts as values
    unique_agreement, counts_agreement = np.unique(agreement, return_counts = True)

    # make two lists of unique candidates and benchmarks that is ordered as unique_and_counts_agreement
    unique_candidate, unique_benchmark = list(zip( *[crosstab_key[u] for u, c in zip(unique_agreement,counts_agreement)] ))

    # make crosstab table
    crosstab_df = pd.DataFrame(data=counts_agreement, index=unique_candidate, columns=unique_benchmark)

    return crosstab_df


def crosstab_ufunc(candidate, benchmark, comparison_func):

    # dimensions of candidates and benchmark must be flattened to 1D prior to this

    agreement_counts = nb.types.Dict.empty(
                            key_type=np.types.unicode_type,
                            value_type=types.float64[:],)
    agreement_encodings = np.types.int64[:]

    for i,(c, b) in enumerate(zip(candidate, benchmark)):
        
        # compute agreement value
        agreement_value = comparison_func(c, b)

        # add to agreement array
        agreement[i] = agreement_value

        # add key to dict and increment or initiate counts
        if (c,b) in agreement_counts:
            agreement_counts[(c,b)] += 1
        else:
            agreement_counts[(c,b)] = 1

    # agreement needs to be reshapen to match candidate and benchmark array dimensions
    # a pairing key needs to be 