![alt text](https://github.com/NOAA-OWP/gval/blob/main/docs/images/gval_dark_mode.png#gh-dark-mode-only) ![alt
text](https://github.com/NOAA-OWP/gval/blob/main/docs/images/gval_light_mode.png#gh-light-mode-only)

GVAL (pronounced "g-val") is a high-level Python framework to evaluate
the skill of geospatial datasets by comparing candidates to benchmark
maps producing agreement maps and metrics.

GVAL is intended to work on raster and vector files as xarray and
geopandas objects, respectively. Abilities to prepare or homogenize maps
for comparison are included. The comparisons are based on scoring
philosophies for three statistical data types including categorical,
continuous, and probabilistic.

Possible enhancements include the ability to trace attributes from the
candidate and benchmark maps to the outputs for analysis purposes.
Additionally, we plan to enable comparisons across a hierarchy of
samples beyond the current map level samples. These sample levels
include catalogs (possibly using [STAC](https://stacspec.org/en)) and
sub-samples (zonal level metrics within maps).

Completed functionality:

- Homogenize candidate and benchmark rasters for comparisons.
- Two-class categorical case for 2 or 3-dimensional rasters.

In-progress:

- Documentation, Github actions, and multi-class categorical raster
    comparisons.

WARNING:

- Our current public API and output formats are likely to change in
    the future.
- Software is provided "AS-IS" without any guarentees. Please QA/QC
    your metrics carefully until this project matures.

# Installation

## General Use

To use this package:

`pip install 'git+https://github.com/NOAA-OWP/gval'`

# Using GVAL

An example of running the entire process for two-class categorical
rasters with one function using minimal arguments is demonstrated below:

``` python
import gval
import rioxarray as rxr

candidate = rxr.open_rasterio('candidate_map_two_class_categorical.tif', mask_and_scale=True)
benchmark = rxr.open_rasterio('benchmark_map_two_class_categorical.tif', mask_and_scale=True)

agreement_map, crosstab_table, metric_table = candidate.gval.categorical_compare(benchmark,
                                                                                 positive_categories=[2],
                                                                                 negative_categories=[0, 1])
```

### Outputs

`agreement_map`

![alt text](https://github.com/NOAA-OWP/gval/blob/main/docs/images/agreement_map.png)

`crosstab_table`

![alt text](https://github.com/NOAA-OWP/gval/blob/main/docs/images/cross_table.png)

`metric_table`

![alt text](https://github.com/NOAA-OWP/gval/blob/main/docs/images/metric_table.png)

For more details on how to use this software, check out this [notebook
tutorial](https://github.com/NOAA-OWP/gval/blob/main/notebooks/Tutorial.ipynb).

# Contributing

Guidelines for contributing to this repository can be found at
[CONTRIBUTING](CONTRIBUTING.MD).

# Citation

Please cite our work if using this package. See 'cite this repository'
in the about section on [GitHub](https://github.com/NOAA-OWP/gval/) or
refer to [CITATION.cff](CITATION.cff)
