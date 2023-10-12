import warnings
from typing import Tuple, Union
from itertools import zip_longest

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import xarray as xr
import contextily as cx
import xyzservices


def _map_plot(
    ds: Union[xr.DataArray, xr.Dataset],
    title: str = "Categorical Map",
    colormap: str = "viridis",
    figsize: Tuple[int, int] = None,
    legend_labels: list = None,
    plot_bands: Union[str, list] = "all",
    plot_type: str = "categorical",
    colorbar_label: Union[str, list] = "",
    basemap: xyzservices.lib.TileProvider = cx.providers.OpenStreetMap.Mapnik,
):
    """
    Plots categorical or continuous Map for xarray object

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
    plot_bands : Union[str, list], default = 'all'
        Which bands to plot if multiple. Default is all bands.
    plot_type : str, default = 'categorical', options
        Whether to plot the map as a categorical map
    color_bar_label : Union[str, list], default =""
        Label or labels for colorbar in the case of continuous plots
    basemap : Union[bool, xyzservices.lib.TileProvider], default = cx.providers.Stamen.Terrain
        Add basemap to the plot

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        QuadMesh Matplotlib object

    Raises
    ------
    ValueError
        Too many values present in dataset for categorical plot.

    References
    ----------
    .. [1] `Matplotlib figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html>`_
    .. [2] `Matplotlib legend <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_
    """

    categorical = True if plot_type == "categorical" else False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Get datasets
        if isinstance(ds, xr.Dataset):
            ds_list = [
                ds[x]
                for x in ds.data_vars
                if plot_bands == "all" or int(x.split("_")[1]) in plot_bands
            ]
            bands = (
                [x + 1 for x in range(len(ds.data_vars))]
                if plot_bands == "all"
                else plot_bands
            )

        else:
            if len(ds.shape) == 3:
                ds_list = [
                    ds.sel({"band": x + 1})
                    for x in range(ds.shape[0])
                    if plot_bands == "all" or x in plot_bands
                ]
                bands = (
                    [x + 1 for x in range(ds.shape[0])]
                    if plot_bands == "all"
                    else plot_bands
                )
            elif len(ds.shape) > 3 or len(ds.shape) < 2:
                raise ValueError("Needs to be 2 or 3 dimensional xarray object")
            else:
                ds_list = [ds]
                bands = [1]

        if len(ds_list) > 8:
            raise ValueError("Cannot plot more than 8 DataArrays at a time")

        cols = 2 if len(ds_list) > 1 else 1
        rows = ((len(ds_list) - 1) // 2) + 1

        if figsize is None:
            figsize = (5 * cols, 4 * rows)

        # Setup figure, axis, and plot
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axes = axs.ravel() if "ravel" in dir(axs) else [axs]

        for i, ax in enumerate(axes):
            if i >= len(ds_list):
                continue

            ds_c = ds_list[i]
            plot = ds_c.plot(ax=ax, cmap=colormap)

            # Print CRS if not large WKT String
            crs = ds_c.rio.crs if len(ds_c.rio.crs) < 25 else ""
            ax.set_title(f"Band {bands[i]}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            if basemap:  # pragma: no cover
                cx.add_basemap(ax, crs=ds_c.rio.crs, source=basemap)

            if categorical:
                # Get colormap values for each unique value
                cmap = matplotlib.colormaps[colormap]
                unique_vals = np.unique(ds_c)

                if len(unique_vals) > 25:
                    raise ValueError(
                        "Too many values present in dataset for categorical plot."
                    )

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
                ax.legend(
                    title="Encodings",
                    handles=legend_elements,
                    bbox_to_anchor=(1.05, 1.0),
                    loc="upper left",
                )

        fig.suptitle(f"{title} ({crs})", fontsize=15)

        if len(ds_list) <= 2:
            top = 0.85
        elif len(ds_list) <= 4:
            top = 0.9
        elif len(ds_list) <= 6:
            top = 0.925
        else:
            top = 0.95

        plt.subplots_adjust(
            top=top, bottom=0.1, left=0.125, right=0.9, hspace=0.6, wspace=0.5
        )

        if categorical:
            # Erase color bar and autoformat x labels to not overlap
            while len(fig.axes) > len(ds_list):
                fig.delaxes(fig.axes[len(ds_list)])
        else:
            # Erase extra axis if present
            if len(ds_list) % 2 == 1 and len(ds_list) > 1:
                fig.delaxes(fig.axes[len(ds_list)])

            use_label = None
            for idx, label in zip_longest(
                range(len(ds_list) // 2, len(ds_list)), colorbar_label
            ):
                # If less labels are provided than graphs apply last label to the remaining graphs
                use_label = use_label if label is None else label
                fig.axes[idx].set_ylabel(label)

        fig.autofmt_xdate()

        fig.show()

        return plot
