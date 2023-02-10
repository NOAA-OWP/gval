"""
Custom exceptions for gval package
"""

__author__ = "Fernando Aristizabal"

class RasterMisalignment(Exception):
    """ Exception raised when rasters dont spatially align. """

    def __init__(self):
        pass

    def __str__(self):
        return f'Rasters do not spatially align.'

class RastersNotSingleBand(Exception):
    """ Exception raised when rasters not single band """

    def __init__(self):
        pass

    def __str__(self):
        return f'Rasters need to be single band'
