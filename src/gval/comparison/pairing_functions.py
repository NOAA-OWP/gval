"""
Comparison functionality
 - Includes pairing and comparison functions
 - crosstabbing functionality
 - Would np.meshgrid lend itself to pairing problem?
TODO:
    - Have not tested parallel case.
    - How to handle xr.Datasets, multiple bands, and multiple variables.
    - consider making a function registry to store pairing functions and their names in a dictionary
        - [Guide to function registration in Python with decorators](https://blog.miguelgrinberg.com/post/the-ultimate-guide-to-python-decorators-part-i-function-registration)
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Iterable, Tuple
from numbers import Number

import numpy as np
import numba as nb


@nb.vectorize(nopython=True)
def _is_not_natural_number(x: Number) -> int:  # pragma: no cover
    """
    Checks value to see if it is a natural number or two non-negative integer [0, 1, 2, 3, 4, ...)

    Parameters
    ----------
    x : Number
        Number to test.

    Returns
    -------
    int
        Return -2 by default. Issue with numba usage. Please ignore for now.
        FIXME: Must return boolean or some other numba type. Having trouble returning none.

    Raises
    ------
    ValueError
        If x not a natural number.
    """
    # checks to make sure it's not a nan value
    if np.isnan(x):
        return -2  # dummy return
    # checks for non-negative and whole number
    elif (x < 0) | ((x - nb.int64(x)) != 0):
        # FIXME: how to print x with message below using numba????
        raise ValueError(
            "Non natural number found (non-negative integers, excluding Inf) [0, 1, 2, 3, 4, ...)"
        )
    # must return something according to signature
    else:
        return -2  # dummy return


@nb.vectorize(nopython=True)
def cantor_pair(c: Number, b: Number) -> Number:  # pragma: no cover
    """
    Produces unique natural number for two non-negative natural numbers (0,1,2,...)

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.

    Returns
    -------
    Number
        Agreement map value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    _is_not_natural_number(c)
    _is_not_natural_number(b)
    return 0.5 * (c**2 + c + 2 * c * b + 3 * b + b**2)


@nb.vectorize(nopython=True)
def szudzik_pair(c: Number, b: Number) -> Number:  # pragma: no cover
    """
    Produces unique natural number for two non-negative natural numbers (0,1,2,3,...).

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.

    Returns
    -------
    Number
        Agreement map value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    _is_not_natural_number(c)
    _is_not_natural_number(b)
    return c**2 + c + b if c >= b else b**2 + c


@nb.vectorize(nopython=True)
def _negative_value_transformation(x: Number) -> Number:  # pragma: no cover
    """
    Transforms negative values for use with pairing functions that only accept non-negative integers.

    Parameters
    ----------
    x : Number
        Negative number to be transformed.

    Returns
    -------
    Number
        Transformed value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    return 2 * x if x >= 0 else -2 * x - 1


@nb.vectorize(nopython=True)
def cantor_pair_signed(c: Number, b: Number) -> Number:  # pragma: no cover
    """
    Output unique natural number for each unique combination of whole numbers using Cantor signed method.

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.

    Returns
    -------
    Number
        Agreement map value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    ct = _negative_value_transformation(c)
    bt = _negative_value_transformation(b)
    return cantor_pair(ct, bt)


@nb.vectorize(nopython=True)
def szudzik_pair_signed(c: Number, b: Number) -> Number:  # pragma: no cover
    """
    Output unique natural number for each unique combination of whole numbers using Szudzik signed method._summary_

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.

    Returns
    -------
    Number
        Agreement map value.

    References
    ----------
    .. [1] [Cantor and Szudzik Pairing Functions](https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/#signed-szudzik)
    """
    ct = _negative_value_transformation(c)
    bt = _negative_value_transformation(b)
    return szudzik_pair(ct, bt)


def _make_pairing_dict(
    unique_candidate_values: Iterable, unique_benchmark_values: Iterable
) -> dict[Tuple[Number, Number], Number]:
    """
    Creates a dict pairing each unique value in candidate and benchmark arrays.

    Parameters
    ----------
    unique_candidate_values : Iterable
        Unique values in candidate map to create pairing dict with.
    unique_benchmark_values : Iterable
        Unique values in benchmark map to create pairing dict with.

    Returns
    -------
    dict[Tuple[Number, Number], Number]
        Dictionary with keys consisting of unique pairings of candidate and benchmark values with value of agreement map for given pairing.
    """
    from itertools import product

    pairing_dict = {
        k: v
        for v, k in enumerate(product(unique_candidate_values, unique_benchmark_values))
    }

    return pairing_dict


@np.vectorize
def pairing_dict_fn(
    c: Number,
    b: Number,
    pairing_dict: dict[Tuple[Number, Number], Number],  # pragma: no cover
) -> Number:
    """
    Produces a pairing dictionary that produces a unique result for every combination ranging from 256 to the number of combinations.

    Parameters
    ----------
    c : Number
        Candidate map value.
    b : Number
        Benchmark map value.
    pairing_dict : dict[Tuple[Number, Number], Number]
        Dictionary with keys of tuple with (c,b) and value to map agreement value to.

    Returns
    -------
    Number
        Agreement map value.
    """
    return pairing_dict[(c, b)]
