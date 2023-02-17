"""
Comparison functionality
 - Includes pairing and comparison functions
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
from numbers import Number

import numpy as np 
import pandas as pd
from xrspatial.zonal import crosstab
import xarray
#import numba as nb


#########################
# Define comparison functions
# takes two values returns one unique value based on combination
# a pairing function can only take two values. One from candidate and another from benchmark.

"""
Consider implementing cantor and szudzik functions in unsigned form:
https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik
"""

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
    
    References:
        https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik
    """	
    _are_not_natural_numbers(c, b)
    return .5 * (c**2 + c + 2*c*b + 3*b + b**2)

def szudzik_pair(c : int, b : int) -> int: 
    """ Produces unique natural number for two non-negative natural numbers (0,1,2,...)
    
    References:
        https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik
    """	
    _are_not_natural_numbers(c, b)
    return c**2 + c + b if c >= b else b**2 + c

def negative_value_transformation(func):
    """
    Transforms negative values for use with pairing functions that only accept non-negative integers
    
    References:
        https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik
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

    Args:
        c (int): _description_
        b (int): _description_

    Returns:
        int: _description_
    References:
        https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik
    """
    return cantor_pair(c, b)

@negative_value_transformation
def szudzik_pair_signed(c : int, b : int) -> int:
    """
    Output unique natural number for each unique combination of whole numbers using Szudzik signed method.

    Args:
        c (int): _description_
        b (int): _description_

    Returns:
        int: _description_
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
    return pairing_dict[c, b]

####################################
# compare numpy arrays

# should this be a decorator for the second stage functions including crosstab???
def compute_agreement_numpy(candidate : np.ndarray, benchmark : np.ndarray, 
                            comparison_function : callable
                           ) -> np.ndarray:
    """ Computes agreement array from candidate and benchmark given a user provided comparison function """

    if not isinstance(comparison_function,np.ufunc):
        agreement =  np.frompyfunc(comparison_function,2,1)(candidate,benchmark)
    else:
        agreement = comparison_ufunc(candidate, benchmark)

    return agreement