"""
Test cases for test_accessors.py
"""

import pandas as pd
from pytest_cases import parametrize

from tests.conftest import _load_xarray, _load_gpkg
from gval.utils.exceptions import RasterMisalignment


candidate_maps = [
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True).sel(
        band=1, drop=True
    ),
]
benchmark_maps = [
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("benchmark_map_0_accessor.tif", mask_and_scale=True).sel(
        band=1, drop=True
    ),
]
candidate_datasets = [
    _load_xarray(
        "candidate_map_0_accessor.tif", mask_and_scale=True, band_as_variable=True
    )
]
benchmark_datasets = [
    _load_xarray(
        "benchmark_map_0_accessor.tif", mask_and_scale=True, band_as_variable=True
    )
]

positive_cat = [2, 2]
negative_cat = [[0, 1], [0, 1]]


@parametrize(
    "candidate_map, benchmark_map, positive_categories, negative_categories",
    list(zip(candidate_maps, benchmark_maps, positive_cat, negative_cat)),
)
def case_data_array_accessor_success(
    candidate_map, benchmark_map, positive_categories, negative_categories
):
    return (candidate_map, benchmark_map, positive_categories, negative_categories)


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
    return (candidate_map, benchmark_map, positive_categories, negative_categories)


@parametrize(
    "candidate_map, benchmark_map", list(zip(candidate_maps[0:1], benchmark_maps[0:1]))
)
def case_data_array_accessor_spatial_alignment(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


@parametrize(
    "candidate_map, benchmark_map", list(zip(candidate_maps[0:1], benchmark_maps[0:1]))
)
def case_data_array_accessor_compute_agreement(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


@parametrize(
    "candidate_map, benchmark_map",
    list(zip(candidate_maps, benchmark_maps)),
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
    return (candidate_map, benchmark_map, exception)


@parametrize(
    "candidate_map, benchmark_map, positive_categories, negative_categories",
    list(
        zip(
            candidate_datasets, benchmark_datasets, positive_cat[0:1], negative_cat[0:1]
        )
    ),
)
def case_data_set_accessor_success(
    candidate_map, benchmark_map, positive_categories, negative_categories
):
    return candidate_map, benchmark_map, positive_categories, negative_categories


@parametrize(
    "candidate_map, benchmark_map", list(zip(candidate_datasets, benchmark_datasets))
)
def case_data_set_accessor_spatial_alignment(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


@parametrize(
    "candidate_map, benchmark_map", list(zip(candidate_datasets, benchmark_datasets))
)
def case_data_set_accessor_compute_agreement(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


@parametrize(
    "candidate_map, benchmark_map",
    list(zip(candidate_datasets, benchmark_datasets)),
)
def case_data_set_accessor_crosstab_table_success(candidate_map, benchmark_map):
    return candidate_map, benchmark_map


exceptions = [RasterMisalignment]


@parametrize(
    "candidate_map, benchmark_map, exception",
    list(zip(candidate_maps, benchmark_maps, exceptions)),
)
def case_data_set_accessor_crosstab_table_fail(candidate_map, benchmark_map, exception):
    return candidate_map, benchmark_map, exception


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
    "candidate_map, benchmark_map, rasterize_attributes",
    [(candidate_maps[0], gdf, ["category"])],
)
def case_data_frame_rasterize_vector_success(
    candidate_map, benchmark_map, rasterize_attributes
):
    return candidate_map, benchmark_map, rasterize_attributes


@parametrize(
    "candidate_map, benchmark_map, rasterize_attributes",
    [(candidate_maps[0], crosstab, ["category"])],
)
def case_data_frame_rasterize_vector_fail(
    candidate_map, benchmark_map, rasterize_attributes
):
    return candidate_map, benchmark_map, rasterize_attributes
