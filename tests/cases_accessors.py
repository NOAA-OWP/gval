"""
Test cases for test_accessors.py
"""

import numpy as np
import pandas as pd
from pytest_cases import parametrize

from tests.conftest import _load_xarray, _load_gpkg
from gval.utils.exceptions import RasterMisalignment


candidate_maps = [
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True).sel(
        band=1, drop=True
    ),
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True),
]
benchmark_maps = [
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True).sel(
        band=1, drop=True
    ),
    _load_gpkg("polygons_two_class_categorical.gpkg"),
]
candidate_datasets = [
    _load_xarray(
        "candidate_map_0_accessor.tif", mask_and_scale=True, band_as_variable=True
    ),
    _load_xarray(
        "candidate_map_0_accessor.tif", mask_and_scale=True, band_as_variable=True
    ),
]
benchmark_datasets = [
    _load_xarray(
        "benchmark_map_0_accessor.tif", mask_and_scale=True, band_as_variable=True
    ),
    _load_gpkg("polygons_two_class_categorical.gpkg"),
]

positive_cat = np.array([2, 2, 2])
negative_cat = np.array([[0, 1], [0, 1], [0, 1]])
rasterize_attrs = [None, None, ["category"]]


@parametrize(
    "candidate_map, benchmark_map, positive_categories, negative_categories, rasterize_attributes",
    list(
        zip(candidate_maps, benchmark_maps, positive_cat, negative_cat, rasterize_attrs)
    ),
)
def case_data_array_accessor_success(
    candidate_map,
    benchmark_map,
    positive_categories,
    negative_categories,
    rasterize_attributes,
):
    return (
        candidate_map,
        benchmark_map,
        positive_categories,
        negative_categories,
        rasterize_attributes,
    )


@parametrize(
    "candidate_map, benchmark_map, positive_categories, negative_categories",
    list(
        zip(
            candidate_maps[0:1],
            benchmark_datasets[0:1],
            positive_cat[0:1],
            negative_cat[0:1],
        )
    ),
)
def case_data_array_accessor_fail(
    candidate_map, benchmark_map, positive_categories, negative_categories
):
    return candidate_map, benchmark_map, positive_categories, negative_categories


@parametrize(
    "candidate_map, benchmark_map, rasterize_attributes",
    list(
        zip(
            [candidate_maps[0], candidate_maps[2]],
            [benchmark_maps[0], benchmark_maps[2]],
            [rasterize_attrs[0], rasterize_attrs[2]],
        )
    ),
)
def case_data_array_accessor_homogenize(
    candidate_map, benchmark_map, rasterize_attributes
):
    return candidate_map, benchmark_map, rasterize_attributes


@parametrize(
    "candidate_map, benchmark_map", list(zip(candidate_maps[0:1], benchmark_maps[0:1]))
)
def case_data_array_accessor_compute_agreement(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


@parametrize(
    "candidate_map, benchmark_map",
    list(zip(candidate_maps[:2], benchmark_maps[:2])),
)
def case_data_array_accessor_crosstab_table_success(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


exceptions = [IndexError, IndexError]


@parametrize(
    "candidate_map, benchmark_map, exception",
    list(zip(candidate_maps, benchmark_maps, exceptions)),
)
def case_data_array_accessor_crosstab_table_fail(
    candidate_map, benchmark_map, exception
):
    return candidate_map, benchmark_map, exception


@parametrize(
    "candidate_map, benchmark_map, positive_categories, negative_categories, rasterize_attributes",
    list(
        zip(
            candidate_datasets,
            benchmark_datasets,
            positive_cat[0:2],
            negative_cat[0:2],
            [rasterize_attrs[0], rasterize_attrs[2]],
        )
    ),
)
def case_data_set_accessor_success(
    candidate_map,
    benchmark_map,
    positive_categories,
    negative_categories,
    rasterize_attributes,
):
    return (
        candidate_map,
        benchmark_map,
        positive_categories,
        negative_categories,
        rasterize_attributes,
    )


@parametrize(
    "candidate_map, benchmark_map, rasterize_attributes",
    list(
        zip(
            candidate_datasets,
            benchmark_datasets,
            [rasterize_attrs[0], rasterize_attrs[2]],
        )
    ),
)
def case_data_set_accessor_homogenize(
    candidate_map, benchmark_map, rasterize_attributes
):
    return candidate_map, benchmark_map, rasterize_attributes


@parametrize(
    "candidate_map, benchmark_map",
    list(zip(candidate_datasets[0:1], benchmark_datasets[0:1])),
)
def case_data_set_accessor_compute_agreement(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


@parametrize(
    "candidate_map, benchmark_map",
    list(zip(candidate_datasets[0:1], benchmark_datasets[0:1])),
)
def case_data_set_accessor_crosstab_table_success(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


exceptions = [RasterMisalignment]


@parametrize(
    "candidate_map, benchmark_map",
    list(zip(candidate_datasets[0:1], benchmark_datasets[0:1])),
)
def case_data_set_accessor_crosstab_table_fail(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


crosstab = pd.DataFrame(
    {
        "band": [1, 1, 1, 1],
        "candidate_values": [1.0, 2.0, 1.0, 2.0],
        "benchmark_values": [0.0, 0.0, 2.0, 2.0],
        "counts": [234234, 23434, 4343, 34343],
    }
)


@parametrize(
    "crosstab_df, positive_categories, negative_categories",
    list(zip([crosstab] * 2, positive_cat, negative_cat)),
)
def case_data_frame_accessor_compute_metrics(
    crosstab_df, positive_categories, negative_categories
):
    return crosstab_df, positive_categories, negative_categories


gdf = _load_gpkg("polygons_two_class_categorical.gpkg")


@parametrize(
    "candidate_map, crs",
    list(
        zip(
            [candidate_maps[0]] * 2,
            [
                "EPSG:5070",
                """PROJCS["NAD27 / California zone II",
                                       GEOGCS["GCS_North_American_1927",DATUM["D_North_American_1927",
                                       SPHEROID["Clarke_1866",6378206.4,294.9786982138982]],
                                       PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],
                                       PROJECTION["Lambert_Conformal_Conic_2SP"],
                                       PARAMETER["standard_parallel_1",39.83333333333334],
                                       PARAMETER["standard_parallel_2",38.33333333333334],
                                       PARAMETER["latitude_of_origin",37.66666666666666],
                                       PARAMETER["central_meridian",-122],
                                       PARAMETER["false_easting",2000000],
                                       PARAMETER["false_northing",0],UNIT["Foot_US",0.30480060960121924]]""",
            ],
        )
    ),
)
def case_data_array_accessor_categorical_plot_success(candidate_map, crs):
    return candidate_map, crs


@parametrize(
    "candidate_map, legend_labels, num_classes",
    list(zip([candidate_maps[0]] * 2, [None, ["a", "b", "c"]], [30, 2])),
)
def case_data_array_accessor_categorical_plot_fail(
    candidate_map, legend_labels, num_classes
):
    return candidate_map, legend_labels, num_classes
