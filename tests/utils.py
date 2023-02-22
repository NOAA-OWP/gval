"""
General utilities for testing sub-package
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import Union

import os


def _set_aws_environment_variables(AWS_KEYS: Union[os.PathLike, str]) -> None:
    """
    Sets AWS keys as environment variables for use with testing functionality.

    Parameters
    ----------
    AWS_KEYS : Union[os.PathLike, str]
        File path to AWS keys.
    """
    with open(AWS_KEYS, "r") as file_handle:
        keys = list(file_handle.readlines())[0].rstrip()
        (
            os.environ["AWS_ACCESS_KEY_ID"],
            os.environ["AWS_SECRET_ACCESS_KEY"],
        ) = keys.split(",")
