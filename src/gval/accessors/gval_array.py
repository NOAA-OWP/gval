from typing import Tuple

import xarray as xr

from gval.accessors.gval_xarray import GVALXarray
from gval.utils.visualize import categorical_plot


@xr.register_dataarray_accessor("gval")
class GVALArray(GVALXarray):
    """
    Class for extending xarray DataArray functionality

    Attributes
    ----------
    _obj : xr.DataArray
       Object to use off the accessor
    data_type : type
        Data type of _obj
    """

    def __init__(self, xarray_obj: xr.DataArray):
        super().__init__(xarray_obj)

    def cat_plot(
        self,
        title: str = "Categorical Map",
        colormap: str = "viridis",
        figsize: Tuple[int, int] = (6, 4),
        legend_labels: list = None,
    ):
        """
        Plots categorical Map for xarray dataset

        Parameters
        __________
        title : str
            Title of map, default = "Categorical Map"
        colormap : str, default = "viridis"
            Colormap of data
        figsize : tuple[int, int], default=(6, 4)
            Size of the plot
        legend_labels : list, default = None
            Override labels in legend

        References
        ----------
        .. [1] [Matplotlib figure](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html)
        .. [2] [Matplotlib legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)
        """

        return categorical_plot(
            self._obj,
            title=title,
            colormap=colormap,
            figsize=figsize,
            legend_labels=legend_labels,
        )
