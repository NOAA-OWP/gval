# Using GVAL

An example of running the entire process with one command using minimal arguments is demonstrated below:

```python
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

![alt text](../images/agreement_map.png)

`cross_table`

![alt text](../images/cross_table.png)

`metric_table`

![alt text](../images/metric_table.png)

For more details on how to use this software, check out this
[notebook tutorial](https://github.com/NOAA-OWP/gval/blob/main/notebooks/Tutorial.ipynb).
