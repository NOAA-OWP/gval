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
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True, chunks="auto"),
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True),
]
benchmark_maps = [
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True).sel(
        band=1, drop=True
    ),
    _load_gpkg("polygons_two_class_categorical.gpkg"),
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True, chunks="auto"),
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True),
]
candidate_datasets = [
    _load_xarray(
        "candidate_map_0_accessor.tif", mask_and_scale=True, band_as_variable=True
    ),
    _load_xarray(
        "candidate_map_0_accessor.tif", mask_and_scale=True, band_as_variable=True
    ),
    _load_xarray(
        "candidate_map_0_accessor.tif",
        mask_and_scale=True,
        band_as_variable=True,
        chunks="auto",
    ),
]
benchmark_datasets = [
    _load_xarray(
        "benchmark_map_0_accessor.tif", mask_and_scale=True, band_as_variable=True
    ),
    _load_gpkg("polygons_two_class_categorical.gpkg"),
    _load_xarray(
        "benchmark_map_0_accessor.tif",
        mask_and_scale=True,
        band_as_variable=True,
        chunks="auto",
    ),
]
plot_maps = [
    _load_xarray("categorical_multiband_4.tif", mask_and_scale=True),
    _load_xarray("categorical_multiband_6.tif", mask_and_scale=True),
    _load_xarray(
        "categorical_multiband_8.tif", mask_and_scale=True, band_as_variable=True
    ).drop_vars("band_8"),
    _load_xarray("categorical_multiband_10.tif", mask_and_scale=True),
    _load_xarray("categorical_multiband_4.tif", mask_and_scale=True).sel(
        {"band": [1, 2, 3]}
    ),
    _load_xarray("categorical_multiband_4.tif", mask_and_scale=True).sel(
        {"band": 1, "x": -169463.7041}
    ),
]


positive_cat = np.array([2, 2, 2, 2, 2, 2])
negative_cat = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
rasterize_attrs = [None, None, ["category"], None, None, None]
memory_strategy = ["normal", "normal", "normal", "normal", "moderate", "aggressive"]
exception_list = [OSError, ValueError, TypeError]


@parametrize(
    "candidate_map, benchmark_map, positive_categories, negative_categories, rasterize_attributes, memory_strategies",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            positive_cat,
            negative_cat,
            rasterize_attrs,
            memory_strategy,
        )
    ),
)
def case_data_array_accessor_success(
    candidate_map,
    benchmark_map,
    positive_categories,
    negative_categories,
    rasterize_attributes,
    memory_strategies,
):
    return (
        candidate_map,
        benchmark_map,
        positive_categories,
        negative_categories,
        rasterize_attributes,
        memory_strategies,
    )


@parametrize(
    "candidate_map, benchmark_map, positive_categories, negative_categories, memory_strategies, exception",
    list(
        zip(
            [candidate_maps[0], candidate_maps[1], candidate_maps[0]],
            [benchmark_maps[0], benchmark_maps[1], benchmark_datasets[0]],
            positive_cat[0:3],
            negative_cat[0:3],
            ["moderate", "arb_error", "normal"],
            exception_list,
        )
    ),
)
def case_data_array_accessor_fail(
    candidate_map,
    benchmark_map,
    positive_categories,
    negative_categories,
    memory_strategies,
    exception,
):
    return (
        candidate_map,
        benchmark_map,
        positive_categories,
        negative_categories,
        memory_strategies,
        exception,
    )


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
    "candidate_map, benchmark_map, vectorized",
    list(
        zip(
            [candidate_maps[0], candidate_maps[0]],
            [benchmark_maps[0], benchmark_maps[0]],
            [False, True],
        )
    ),
)
def case_data_array_accessor_compute_agreement(
    candidate_map, benchmark_map, vectorized
):
    return candidate_map, benchmark_map, vectorized


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
            positive_cat[0:3],
            negative_cat[0:3],
            [rasterize_attrs[0], rasterize_attrs[2], rasterize_attrs[0]],
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
    "candidate_map, crs, entries",
    list(
        zip(
            [
                candidate_maps[0],
                candidate_maps[1],
                plot_maps[0],
                plot_maps[1],
                plot_maps[2],
            ],
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
                "EPSG:5070",
                "EPSG:5070",
                "EPSG:5070",
            ],
            [2, 2, 3, 3, 4],
        )
    ),
)
def case_categorical_plot_success(candidate_map, crs, entries):
    return candidate_map, crs, entries


@parametrize(
    "candidate_map, legend_labels, num_classes",
    list(
        zip(
            [candidate_maps[0], candidate_maps[0], plot_maps[3]],
            [None, ["a", "b", "c"], ["a", "b"]],
            [30, 2, 2],
        )
    ),
)
def case_categorical_plot_fail(candidate_map, legend_labels, num_classes):
    return candidate_map, legend_labels, num_classes


@parametrize(
    "candidate_map, axes",
    list(zip([candidate_maps[0], plot_maps[4]], [2, 6])),
)
def case_continuous_plot_success(candidate_map, axes):
    return candidate_map, axes


@parametrize(
    "candidate_map",
    [plot_maps[3], plot_maps[5]],
)
def case_continuous_plot_fail(candidate_map):
    return candidate_map


candidate_maps = ["candidate_continuous_0.tif", "candidate_continuous_1.tif"]
benchmark_maps = ["benchmark_continuous_0.tif", "benchmark_continuous_1.tif"]


@parametrize(
    "candidate_map, benchmark_map",
    list(zip(candidate_maps, benchmark_maps)),
)
def case_data_array_accessor_continuous(candidate_map, benchmark_map):
    return _load_xarray(candidate_map), _load_xarray(benchmark_map)


@parametrize(
    "candidate_map, benchmark_map",
    list(zip(candidate_maps, benchmark_maps)),
)
def case_data_set_accessor_continuous(candidate_map, benchmark_map):
    return (
        _load_xarray(candidate_map, band_as_variable=True),
        _load_xarray(benchmark_map, band_as_variable=True),
    )
