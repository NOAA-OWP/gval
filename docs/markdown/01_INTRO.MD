![alt text](../images/gval_dark_mode.png#gh-dark-mode-only)
![alt text](../images/gval_light_mode.png#gh-light-mode-only)

![Build and Test](https://github.com/NOAA-OWP/gval/actions/workflows/python-app.yml/badge.svg)![Coverage](https://github.com/NOAA-OWP/gval/raw/main/docs/images/coverage.svg)

GVAL (pronounced "g-val") is a high-level Python framework to evaluate the skill of geospatial datasets by comparing candidates to benchmark maps producing agreement maps and metrics.

GVAL is intended to work on raster and vector files as xarray and geopandas objects, respectively. Abilities to prepare or homogenize maps for comparison are included. The comparisons are based on scoring philosophies for three statistical data types including categorical, continuous, and probabilistic.

WARNING:
- Our current public API and output formats are likely to change in the future.
- Software is provided "AS-IS" without any guarantees. Please QA/QC your metrics carefully until this project matures.
