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
from functools import partial

import numpy as np
import numba as nb


@nb.vectorize(nopython=True)
def _is_not_natural_number(
    x: Number, raise_exception: bool
) -> bool:  # pragma: no cover
    """
    Checks value to see if it is a natural number or two non-negative integer [0, 1, 2, 3, 4, ...)

    Parameters
    ----------
    x : Number
        Number to test.
    raise_exception : bool
        Raise exception or return bool.

    Returns
    -------
    bool
        Is natural number or not. np.nan treated as natural.

    Raises
    ------
    ValueError
        If x not a natural number.
    """
    # checks to make sure it's not a nan value
    if np.isnan(x):
        return False  # treated as natural for this use case

    # checks for non-negative and whole number
    elif (x < 0) | ((x - nb.int64(x)) != 0):
        if raise_exception:
            raise ValueError(
                "Non natural number found (non-negative integers, excluding Inf) [0, 1, 2, 3, 4, ...)"
            )
        else:
            return True

    # is natural
    else:
        return False


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
    _is_not_natural_number(c, True)
    _is_not_natural_number(b, True)
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
    _is_not_natural_number(c, True)
    _is_not_natural_number(b, True)

    if np.isnan(c) or np.isnan(b):
        return np.nan
    else:
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
    if np.isnan(x):
        return x
    else:
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

    if np.isnan(c) or np.isnan(b):
        return np.nan
    else:
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

    if np.isnan(c) or np.isnan(b):
        return np.nan
    else:
        return szudzik_pair(ct, bt)


class PairingDict(dict):
    """
    Pairing dictionary class that enables to replace np.nans with a different value as np.nan != np.nan.

    Parameters
    ----------
    dict : dict
        Parent class.

    Attributes
    -------
    replacement_value
        Value to use instead of np.nan.
    """

    # TODO: Perhaps this could be np.finfo(float).max
    replacement_value = "NaN"

    def __init__(self, *args, **kwargs):
        # make a tmp dict with
        tmp_dict = dict(*args, **kwargs)

        # initialize self
        super().__init__()

        # set the items
        for k, v in tmp_dict.items():
            self.__setitem__(k, v)

    def _replace_nans(self, key):
        """Replaces NaNs"""
        new_key = [None] * len(key)
        for i, k in enumerate(key):
            if k == self.replacement_value:
                new_key[i] = k
            elif np.isnan(k):  # cannot be a string
                new_key[i] = self.replacement_value
            else:
                new_key[i] = k

        return tuple(new_key)

    def __getitem__(self, key):
        key = self._replace_nans(key)
        # FIXME:  Returns nans when present in key but any other key in the mean time
        if self.replacement_value in key:
            return np.nan
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        key = self._replace_nans(key)
        super().__setitem__(key, value)

    def __contains__(self, key):  # pragma: no cover
        # FIXME: Why are these two lines not covered? pragma: no cover for the time being
        key = self._replace_nans(key)
        return super().__contains__(key)


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

    # return pairing_dict
    return PairingDict(pairing_dict)


vectorize_partial = partial(np.vectorize, otypes=[float])


# REVALUATE Performance
@vectorize_partial
def pairing_dict_fn(
    c: Number,
    b: Number,
    pairing_dict: dict[Tuple[Number, Number], Number],
) -> Number:  # pragma: no cover
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
