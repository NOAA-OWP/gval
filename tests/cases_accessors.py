"""
Test cases for test_accessors.py
"""

import numpy as np
import pandas as pd
import xarray as xr
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

agreement_maps = [
    _load_xarray("agreement_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("agreement_map_0_accessor.tif", mask_and_scale=True).sel(
        band=1, drop=True
    ),
    _load_xarray(
        "agreement_map_1_accessor.tif", mask_and_scale=True, band_as_variable=True
    ),
]
agreement_map_fail = [
    _load_xarray("agreement_accessor_fail.tif", mask_and_scale=True),
    _load_xarray(
        "agreement_accessor_fail.tif", mask_and_scale=True, band_as_variable=True
    ),
]

positive_cat = np.array([2, 2, 2, 2, 2, 2, 2])
negative_cat = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
rasterize_attrs = [None, None, ["category"], None, None, None, None]
memory_strategy = [
    "normal",
    "normal",
    "normal",
    "normal",
    "moderate",
    "aggressive",
    "normal",
]
exception_list = [OSError, ValueError, TypeError]
comparison_funcs = [
    "szudzik",
    "szudzik",
    "szudzik",
    "szudzik",
    "szudzik",
    "szudzik",
    "pairing_dict",
]


@parametrize(
    "candidate_map, benchmark_map, positive_categories, negative_categories, rasterize_attributes, memory_strategies, comparison_function",
    list(
        zip(
            [
                candidate_maps[0],
                candidate_maps[1],
                candidate_maps[2],
                candidate_maps[3],
                candidate_maps[4],
                candidate_maps[5],
                candidate_maps[0],
            ],
            [
                benchmark_maps[0],
                benchmark_maps[1],
                benchmark_maps[2],
                benchmark_maps[3],
                benchmark_maps[4],
                benchmark_maps[5],
                benchmark_maps[0],
            ],
            positive_cat,
            negative_cat,
            rasterize_attrs,
            memory_strategy,
            comparison_funcs,
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
    comparison_function,
):
    return (
        candidate_map,
        benchmark_map,
        positive_categories,
        negative_categories,
        rasterize_attributes,
        memory_strategies,
        comparison_function,
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
    "agreement_map",
    list(agreement_maps[:2]),
)
def case_data_array_accessor_crosstab_table_success(agreement_map):
    return agreement_map


exceptions = [KeyError]


@parametrize(
    "agreement_map, exception",
    list(zip(agreement_map_fail[:1], exceptions)),
)
def case_data_array_accessor_crosstab_table_fail(agreement_map, exception):
    return agreement_map, exception


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
    "agreement_map",
    list(agreement_maps[2:]),
)
def case_data_set_accessor_crosstab_table_success(agreement_map):
    return agreement_map


exceptions = [RasterMisalignment]


@parametrize(
    "agreement_map",
    list(agreement_map_fail[1:]),
)
def case_data_set_accessor_crosstab_table_fail(agreement_map):
    return agreement_map


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


@parametrize(
    "vector_map, reference_map, attributes",
    list(zip(benchmark_maps[2:3], candidate_maps[0:1], [["category"]])),
)
def case_dataframe_accessor_rasterize(vector_map, reference_map, attributes):
    return vector_map, reference_map, attributes


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


candidate_maps = ["candidate_map_attrs_1.tif"]
benchmark_maps = ["benchmark_map_attrs_1.tif"]
agreement_maps = [xr.DataArray(np.ones((3, 3)), dims=["y", "x"])]


@parametrize(
    "candidate_map, benchmark_map, agreement_map",
    list(zip(candidate_maps, benchmark_maps, agreement_maps)),
)
def case_accessor_attributes(candidate_map, benchmark_map, agreement_map):
    return _load_xarray(candidate_map), _load_xarray(benchmark_map), agreement_map
