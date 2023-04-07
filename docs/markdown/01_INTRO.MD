![alt text](../images/gval_dark_mode.png#gh-dark-mode-only)
![alt text](../images/gval_light_mode.png#gh-light-mode-only)

GVAL (pronounced "g-val") is a high-level Python framework to evaluate the skill of geospatial datasets by comparing candidates to benchmark maps producing agreement maps and metrics.

GVAL is intended to work on raster and vector files as xarray and geopandas objects, respectively. Abilities to prepare or homogenize maps for comparison are included. The comparisons are based on scoring philosophies for three statistical data types including categorical, continuous, and probabilistic.

Possible enhancements include the ability to trace attributes from the candidate and benchmark maps to the outputs for analysis purposes. Additionally, we plan to enable comparisons across a hierarchy of samples beyond the current map level samples. These sample levels include catalogs (possibly using [STAC](https://stacspec.org/en)) and sub-samples (zonal level metrics within maps).

Completed functionality:
- Homogenize candidate and benchmark rasters for comparisons.
- Two-class categorical case for 2 or 3-dimensional rasters.

In-progress:
- Documentation, Github actions, and multi-class categorical raster comparisons.

WARNING: Our current public API and output formats are likely to change in the future.
