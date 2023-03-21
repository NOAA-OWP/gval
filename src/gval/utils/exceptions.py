"""
Custom exceptions for gval package
"""

__author__ = "Fernando Aristizabal"


class RasterMisalignment(Exception):  # pragma: no cover
    """Exception raised when rasters don't spatially align."""

    def __init__(self):
        pass

    def __str__(self):
        return "Rasters do not spatially align."


class RastersDontIntersect(Exception):  # pragma: no cover
    """Exception raised when rasters don't spatially intersect."""

    def __init__(self):
        pass

    def __str__(self):
        return "Rasters don't spatially intersect."
