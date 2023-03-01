"""
General utilities for testing sub-package
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np


def _assert_pairing_dict_equal(computed_dict: dict, expected_dict: dict) -> None:
    """
    Testing function used to test if two pairing dictionaries are equal.

    This is necessary because np.nans can be of float or np.float64 kind which makes operator (==) comparisons false.

    Parameters
    ----------
    computed_dict : dict
        Pairing dict computed to test.
    expected_dict : dict
        Expected pairing dict to compare to.

    Returns
    -------
    None

    See also
    --------
    :obj:`np.testing.assert_equal`

    Raises
    ------
    AssertionError
    """

    np.testing.assert_equal(list(computed_dict.keys()), list(expected_dict.keys()))
    np.testing.assert_equal(list(computed_dict.values()), list(expected_dict.values()))
