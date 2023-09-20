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


if __name__ == "__main__":
    import geopandas as gpd

    data_path = "/home/sven/repos/gval/notebooks"
    polygons_include = gpd.read_file(f"{data_path}/subsample_continuous_polygons.gpkg")
    print(polygons_include.to_dict())
