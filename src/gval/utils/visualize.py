import warnings
from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import xarray as xr


def categorical_plot(
    ds: xr.DataArray,
    title: str = "Categorical Map",
    colormap: str = "viridis",
    figsize: Tuple[int, int] = (6, 4),
    legend_labels: list = None,
):
    """
    Plots categorical Map for xarray dataset

    Parameters
    __________
    ds : xr.DataArray
        Dataset to create categorical map
    title : str
        Title of map, default = "Categorical Map"
    colormap : str, default = "viridis"
        Colormap of data
    figsize : tuple[int, int], default=None
        Size of the plot
    legend_labels : list, default = None
        Override labels in legend

    Returns
    -------
    QuadMesh Matplotlib object

    Raises
    ------
    ValueError
        Too many values present in dataset for categorical plot.

    References
    ----------
    .. [1] [Matplotlib figure](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html)
    .. [2] [Matplotlib legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Setup figure, axis, and plot
        fig, ax1 = plt.subplots(figsize=figsize)
        plot = ds.plot(ax=ax1, cmap=colormap)

        # Print CRS if not large WKT String
        crs = ds.rio.crs if len(ds.rio.crs) < 25 else ""
        ax1.set_title(f"{title} ({crs})")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")

        # Get colormap values for each unique value
        cmap = matplotlib.colormaps[colormap]
        unique_vals = np.unique(ds)

        if len(unique_vals) > 25:
            raise ValueError("Too many values present in dataset for categorical plot.")

        if legend_labels is not None and len(legend_labels) != len(unique_vals):
            raise ValueError("Need as many labels as unique values.")

        # Setup normalized color ramp
        norm = matplotlib.colors.Normalize(
            vmin=np.nanmin(unique_vals), vmax=np.nanmax(unique_vals)
        )

        # Create legend
        labels = unique_vals if legend_labels is None else legend_labels
        legend_elements = [
            Patch(color=cmap(norm(val)), label=str(label))
            for val, label in zip(unique_vals, labels)
            if not np.isnan(val)
        ]
        ax1.legend(
            title="Encodings",
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
        )

        # Erase color bar and autoformat x labels to not overlap
        fig.delaxes(fig.axes[1])
        fig.autofmt_xdate()

        fig.show()

        return plot
