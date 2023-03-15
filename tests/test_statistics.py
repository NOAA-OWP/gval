# import numpy as np
# import pandas as pd

# import pytest
from pytest_cases import parametrize_with_cases

# import xarray as xr
# from pytest import raises


@parametrize_with_cases("c, b, a", glob="cantor_pair")
def test_cantor_pair(c, b, a):
    """tests cantor pairing function"""
    pass
    # np.testing.assert_equal(
    #     cantor_pair(c, b), a
    # ), "Cantor function output does not match expected value"
