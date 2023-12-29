 ![alt
text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/gval_light_mode.png#gh-light-mode-only)

![Build and
Test](https://github.com/NOAA-OWP/gval/actions/workflows/python-app.yml/badge.svg)![Coverage](https://github.com/NOAA-OWP/gval/raw/testing/docs/images/coverage.svg)[![PyPI
version](https://badge.fury.io/py/gval.svg)](https://badge.fury.io/py/gval)

GVAL (pronounced "g-val") is a high-level Python framework to evaluate
the skill of geospatial datasets by comparing candidates to benchmark
maps producing agreement maps and metrics.

GVAL is intended to work on raster and vector files as xarray and
geopandas objects, respectively. Abilities to prepare or homogenize maps
for comparison are included. The comparisons are based on scoring
philosophies for three statistical data types including categorical,
continuous, and probabilistic.

See the full [documentation](https://noaa-owp.github.io/gval/).

WARNING:

- Our current public API and output formats are likely to change in the
  future.
- Software is provided "AS-IS" without any guarantees. Please QA/QC your
  metrics carefully until this project matures.

# Installation

## General Use

To use this package:

`pip install gval`

Or for bleeding edge updates install from the repository:

`pip install 'git+https://github.com/NOAA-OWP/gval'`

# Using GVAL

### Categorical Example

An example of running the entire process for two-class categorical
rasters with one function using minimal arguments is demonstrated below:

``` python
import gval
import rioxarray as rxr

candidate = rxr.open_rasterio('candidate_map_two_class_categorical.tif', mask_and_scale=True)
benchmark = rxr.open_rasterio('benchmark_map_two_class_categorical.tif', mask_and_scale=True)

(agreement_map,
 crosstab_table,
 metric_table) = candidate.gval.categorical_compare(benchmark,
                                                   positive_categories=[2],
                                                   negative_categories=[0, 1])
```

#### Categorical Outputs

`agreement_map`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/agreement_map.png)

`crosstab_table`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/cross_table.png)

`metric_table`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/metric_table.png)

### Continuous Example

The same can be done for rasters with continuous valued statistical data
types as shown below (in this case only a subset of the default
statistics will be run):

``` python
import gval
import rioxarray as rxr

candidate = rxr.open_rasterio('livneh_2011_precip.tif', mask_and_scale=True) # VIC
benchmark = rxr.open_rasterio('prism_2011_precip.tif', mask_and_scale=True) # PRISM

agreement, metric_table = candidate.gval.continuous_compare(
    benchmark,
    metrics=[
        "coefficient_of_determination",
        "mean_percentage_error",
        "mean_absolute_percentage_error",
        "mean_normalized_mean_absolute_error"
    ]
)
```

#### Continuous Outputs

`agreement_map`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/continuous_agreement_map.png)

`metric_table`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/continuous_metric_table.png)

### Catalog Example

Entire catalogs can be compared in GVAL, which are represented by
dataframes of maps. The following is a candidate and benchmark catalog
for continuous datasets:

`candidate_catalog`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/candidate_catalog.png)

`benchmark_catalog`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/benchmark_catalog.png)

With the following code a comparison of each pair of maps can be run
with the following code. Since the parameter `agreement_map_field` is
provided the column `agreement_maps` found in the candidate catalog will
be used to export the agreement map to that location. (Note the first
pair of candidate and benchmark maps are single band rasters while the
second pair are multiband rasters):

``` python
import pandas as pd

from gval.catalogs.catalogs import catalog_compare

candidate_continuous_catalog = pd.read_csv('candidate_catalog_0.csv')
benchmark_continuous_catalog = pd.read_csv('benchmark_catalog_0.csv')

arguments = {
    "candidate_catalog": candidate_continuous_catalog,
    "benchmark_catalog": benchmark_continuous_catalog,
    "on": "compare_id",
    "agreement_map_field": "agreement_maps",
    "map_ids": "map_id",
    "how": "inner",
    "compare_type": "continuous",
    "compare_kwargs": {
        "metrics": (
            "coefficient_of_determination",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
        ),
        "encode_nodata": True,
        "nodata": -9999,
    },
    "open_kwargs": {
        "mask_and_scale": True,
        "masked": True
    }
}

agreement_continuous_catalog = catalog_compare(**arguments)
```

#### Catalog Outputs

`agreement_map`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/catalog_agreement_map.png) ![alt
text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/catalog_agreement_map2.png)

`catalog_metrics`

![alt text](https://github.com/NOAA-OWP/gval/raw/main/docs/images/catalog_metric_table.png)

(Note that both catalog level attributes in the candidate and benchmark
catalogs are present in the catalog metrics table.)

For more detailed examples of how to use this software, check out these
[notebook
tutorials](https://github.com/NOAA-OWP/gval/blob/main/notebooks).

# Contributing

Guidelines for contributing to this repository can be found at
[CONTRIBUTING](https://github.com/NOAA-OWP/gval/blob/main/CONTRIBUTING.MD).

# Citation

Please cite our work if using this package. See 'cite this repository'
in the about section on [GitHub](https://github.com/NOAA-OWP/gval/) or
refer to
[CITATION.cff](https://github.com/NOAA-OWP/gval/blob/main/CITATION.cff)
