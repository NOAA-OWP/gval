"""
Test cases for catalogs module.
"""

# __all__ = ['*']

from pytest_cases import parametrize

import pandas as pd
from pandas import Timestamp
from dask import dataframe as dd
import xarray as xr
import numpy as np
from json import JSONDecodeError

from tests.conftest import TEST_DATA_DIR


# test cases for _compare_catalogs in catalogs.py
candidate_catalogs = [
    pd.DataFrame(
        {
            "map_id": [
                f"{TEST_DATA_DIR}/candidate_continuous_0.tif",
                f"{TEST_DATA_DIR}/candidate_continuous_1.tif",
            ],
            "compare_id": ["compare1", "compare2"],
            "value1": [1, 2],
            "value2": [5, 6],
            "agreement_maps": [
                "agreement_continuous_0.tif",
                "agreement_continuous_1.tif",
            ],
        }
    )
] * 2 + [
    pd.DataFrame(
        {
            "map_id": [
                f"{TEST_DATA_DIR}/candidate_categorical_0.tif",
                f"{TEST_DATA_DIR}/candidate_categorical_1.tif",
            ],
            "compare_id": ["compare1", "compare2"],
            "value1": [1, 2],
            "value2": [5, 6],
            "agreement_maps": [
                "agreement_categorical_0.tif",
                "agreement_categorical_1.tif",
            ],
        }
    )
] * 2

benchmark_catalogs = [
    pd.DataFrame(
        {
            "map_id": [
                f"{TEST_DATA_DIR}/benchmark_continuous_0.tif",
                f"{TEST_DATA_DIR}/benchmark_continuous_1.tif",
            ],
            "compare_id": ["compare1", "compare2"],
            "value1": [1, 2],
            "value2": [5, 6],
        }
    )
] * 2 + [
    pd.DataFrame(
        {
            "map_id": [
                f"{TEST_DATA_DIR}/benchmark_categorical_0.tif",
                f"{TEST_DATA_DIR}/benchmark_categorical_1.tif",
            ],
            "compare_id": ["compare1", "compare2"],
            "value1": [1, 2],
            "value2": [5, 6],
        }
    )
] * 2

expected = [
    pd.DataFrame(
        {
            "map_id_candidate": [
                "s3://gval-test/candidate_continuous_0.tif",
                "s3://gval-test/candidate_continuous_1.tif",
                "s3://gval-test/candidate_continuous_1.tif",
            ],
            "compare_id": ["compare1", "compare2", "compare2"],
            "map_id_benchmark": [
                "s3://gval-test/benchmark_continuous_0.tif",
                "s3://gval-test/benchmark_continuous_1.tif",
                "s3://gval-test/benchmark_continuous_1.tif",
            ],
            "value1_candidate": [1, 2, 2],
            "value2_candidate": [5, 6, 6],
            "agreement_maps": [
                "agreement_continuous_0.tif",
                "agreement_continuous_1.tif",
                "agreement_continuous_1.tif",
            ],
            "value1_benchmark": [1, 2, 2],
            "value2_benchmark": [5, 6, 6],
            "band": [1.0, 1.0, 2.0],
            "coefficient_of_determination": [
                -0.06615996360778809,
                -2.829420804977417,
                0.10903036594390869,
            ],
            "mean_absolute_error": [
                0.3173885941505432,
                0.48503121733665466,
                0.48503121733665466,
            ],
            "mean_absolute_percentage_error": [
                0.15956786274909973,
                0.20223499834537506,
                0.15323485434055328,
            ],
        }
    )
] * 2 + [
    pd.DataFrame(
        {
            "map_id_candidate": [
                f"{TEST_DATA_DIR}/candidate_categorical_0.tif",
                f"{TEST_DATA_DIR}/candidate_categorical_1.tif",
                f"{TEST_DATA_DIR}/candidate_categorical_1.tif",
            ],
            "compare_id": ["compare1", "compare2", "compare2"],
            "map_id_benchmark": [
                f"{TEST_DATA_DIR}/benchmark_categorical_0.tif",
                f"{TEST_DATA_DIR}/benchmark_categorical_1.tif",
                f"{TEST_DATA_DIR}/benchmark_categorical_1.tif",
            ],
            "value1_candidate": [1, 2, 2],
            "value2_candidate": [5, 6, 6],
            "agreement_maps": [
                "agreement_categorical_0.tif",
                "agreement_categorical_1.tif",
                "agreement_categorical_1.tif",
            ],
            "value1_benchmark": [1, 2, 2],
            "value2_benchmark": [5, 6, 6],
            "band": [1.0, 1.0, 2.0],
            "fn": [844.0] * 3,
            "fp": [844.0] * 3,
            "tn": [np.nan] * 3,
            "tp": [1977.0] * 3,
            "critical_success_index": [0.539427] * 3,
            "true_positive_rate": [0.700815] * 3,
            "positive_predictive_value": [0.700815] * 3,
        }
    )
] * 2

on = ["compare_id"] * 4
map_ids = ["map_id", ("map_id", "map_id"), "map_id", "map_id"]
how = ["inner"] * 4
compare_type = ["continuous"] * 2 + ["categorical"] * 2
dask = [False, True] * 2

compare_kwargs = [
    {
        "metrics": (
            "coefficient_of_determination",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
        ),
        "encode_nodata": True,
        "nodata": -9999,
    }
] * 2 + [
    {
        "metrics": (
            "critical_success_index",
            "true_positive_rate",
            "positive_predictive_value",
        ),
        "encode_nodata": True,
        "nodata": -9999,
        "positive_categories": 2,
    },
    {
        "metrics": (
            "critical_success_index",
            "true_positive_rate",
            "positive_predictive_value",
        ),
        "encode_nodata": True,
        "nodata": -9999,
        "positive_categories": 2,
    },
]

open_kwargs = [
    {"mask_and_scale": True, "masked": True},
    {"mask_and_scale": True, "masked": True, "chunks": "auto"},
    {"mask_and_scale": True, "masked": True},
    {"mask_and_scale": True, "masked": True},
]

# agreement_map_field = [None, "agreement_maps"]
agreement_map_field = ["agreement_maps"] * 4
agreement_map_write_kwargs = [{"tiled": True, "windowed": True}] * 4

expected_agreement_maps = [
    (
        f"{TEST_DATA_DIR}/agreement_continuous_0.tif",
        f"{TEST_DATA_DIR}/agreement_continuous_1.tif",
        f"{TEST_DATA_DIR}/agreement_continuous_1.tif",
    )
] * 2 + [
    (
        f"{TEST_DATA_DIR}/agreement_categorical_0.tif",
        f"{TEST_DATA_DIR}/agreement_categorical_1.tif",
        f"{TEST_DATA_DIR}/agreement_categorical_1.tif",
    )
] * 2


@parametrize(
    "candidate_catalog, benchmark_catalog, on, map_ids, how, compare_type, compare_kwargs, open_kwargs, agreement_map_field, dask, expected, expected_agreement_map",
    list(
        zip(
            candidate_catalogs,
            benchmark_catalogs,
            on,
            map_ids,
            how,
            compare_type,
            compare_kwargs,
            open_kwargs,
            agreement_map_field,
            dask,
            expected,
            expected_agreement_maps,
        )
    ),
)
def case_compare_catalogs(
    candidate_catalog,
    benchmark_catalog,
    on,
    map_ids,
    how,
    compare_type,
    compare_kwargs,
    open_kwargs,
    agreement_map_field,
    dask,
    expected,
    expected_agreement_map,
):
    if dask:
        candidate_catalog = dd.from_pandas(candidate_catalog, npartitions=1)
        benchmark_catalog = dd.from_pandas(benchmark_catalog, npartitions=1)

    return (
        candidate_catalog,
        benchmark_catalog,
        on,
        map_ids,
        how,
        compare_type,
        compare_kwargs,
        open_kwargs,
        agreement_map_field,
        expected,
        expected_agreement_map,
    )


compare_kwargs = [None] * 2
open_kwargs = [None] * 2
compare_type = [
    lambda cm, bm: tuple(
        [
            xr.DataArray(np.random.rand(3, 3), coords={"y": [1, 2, 3], "x": [1, 2, 3]}),
            pd.DataFrame([{"agreement_map": ["test"], "map_id": ["test"]}]),
        ]
    )
] * 2
dask = [False, False]


@parametrize(
    "candidate_catalog, benchmark_catalog, on, map_ids, how, compare_type, compare_kwargs, open_kwargs, agreement_map_field, dask, expected, expected_agreement_map",
    list(
        zip(
            candidate_catalogs,
            benchmark_catalogs,
            on,
            map_ids,
            how,
            compare_type,
            compare_kwargs,
            open_kwargs,
            agreement_map_field,
            dask,
            expected,
            expected_agreement_maps,
        )
    ),
)
def case_compare_catalogs_no_kwargs(
    candidate_catalog,
    benchmark_catalog,
    on,
    map_ids,
    how,
    compare_type,
    compare_kwargs,
    open_kwargs,
    agreement_map_field,
    dask,
    expected,
    expected_agreement_map,
):
    if dask:
        candidate_catalog = dd.from_pandas(candidate_catalog, npartitions=1)
        benchmark_catalog = dd.from_pandas(benchmark_catalog, npartitions=1)

    return (
        candidate_catalog,
        benchmark_catalog,
        on,
        map_ids,
        how,
        compare_type,
        compare_kwargs,
        open_kwargs,
        agreement_map_field,
        expected,
        expected_agreement_map,
    )


compare_kwargs = [
    {
        "metrics": (
            "critical_success_index",
            "true_positive_rate",
            "positive_predictive_value",
        ),
        "encode_nodata": True,
        "nodata": -9999,
    }
] * 2

compare_type = ["categorical"] * 2


@parametrize(
    "candidate_catalog, benchmark_catalog, on, map_ids, how, compare_type, compare_kwargs, open_kwargs, agreement_map_field, dask, expected, expected_agreement_map",
    list(
        zip(
            candidate_catalogs,
            benchmark_catalogs,
            on,
            map_ids,
            how,
            compare_type,
            compare_kwargs,
            open_kwargs,
            agreement_map_field,
            dask,
            expected,
            expected_agreement_maps,
        )
    ),
)
def case_compare_catalogs_categorical(
    candidate_catalog,
    benchmark_catalog,
    on,
    map_ids,
    how,
    compare_type,
    compare_kwargs,
    open_kwargs,
    agreement_map_field,
    dask,
    expected,
    expected_agreement_map,
):
    if dask:
        candidate_catalog = dd.from_pandas(candidate_catalog, npartitions=1)
        benchmark_catalog = dd.from_pandas(benchmark_catalog, npartitions=1)

    return (
        candidate_catalog,
        benchmark_catalog,
        on,
        map_ids,
        how,
        compare_type,
        compare_kwargs,
        open_kwargs,
        agreement_map_field,
        expected,
        expected_agreement_map,
    )


map_ids = [1, "map_id", "map_id", "map_id"]
expected_exception = [ValueError, NotImplementedError, ValueError, ValueError]
compare_type = ["continuous", "probabilistic", "test_anything", 1e5]

candidate_catalogs *= 2
benchmark_catalogs *= 2
on *= 2
how *= 2
compare_kwargs *= 2
open_kwargs *= 2
agreement_map_field *= 2
dask *= 2


@parametrize(
    "candidate_catalog, benchmark_catalog, on, map_ids, how, compare_type, compare_kwargs, open_kwargs, agreement_map_field, dask, expected_exception",
    list(
        zip(
            candidate_catalogs,
            benchmark_catalogs,
            on,
            map_ids,
            how,
            compare_type,
            compare_kwargs,
            open_kwargs,
            agreement_map_field,
            dask,
            expected_exception,
        )
    ),
)
def case_compare_catalogs_fail(
    candidate_catalog,
    benchmark_catalog,
    on,
    map_ids,
    how,
    compare_type,
    compare_kwargs,
    open_kwargs,
    agreement_map_field,
    dask,
    expected_exception,
):
    if dask:
        candidate_catalog = dd.from_pandas(candidate_catalog, npartitions=1)
        benchmark_catalog = dd.from_pandas(benchmark_catalog, npartitions=1)

    return (
        candidate_catalog,
        benchmark_catalog,
        on,
        map_ids,
        how,
        compare_type,
        compare_kwargs,
        open_kwargs,
        agreement_map_field,
        expected_exception,
    )


url = "https://earth-search.aws.element84.com/v1"
collection = "sentinel-2-l2a"
times = ["2020-04-01", "2020-04-03"]
bbox = [-105.78, 35.79, -105.72, 35.84]
assets = ["aot"]

expected_stac_df = {
    "collection_id_candidate": ["sentinel-2-l2a", "sentinel-2-l2a"],
    "item_id_candidate": ["S2B_13SDV_20200401_1_L2A", "S2B_13SDV_20200401_0_L2A"],
    "item_time_candidate": [
        Timestamp("2020-04-01 18:04:04.327000+0000", tz="utc"),
        Timestamp("2020-04-01 18:04:04.327000+0000", tz="utc"),
    ],
    "create_time_candidate": ["2023-10-07T12:09:07.273Z", "2022-11-06T10:14:16.681Z"],
    "map_id_candidate": [
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_1_L2A/AOT.tif",
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_0_L2A/AOT.tif",
    ],
    "map_name_candidate": ["aot", "aot"],
    "compare_id": [1, 2],
    "coverage_geometry_type_candidate": ["Polygon", "Polygon"],
    "coverage_geometry_coords_candidate": [
        [
            [
                [-106.11191385205835, 36.13972769324406],
                [-106.0982941692468, 35.150822790058335],
                [-105.85393882465051, 35.15385918076215],
                [-105.68102037327243, 35.69893282033011],
                [-105.59450557216445, 35.97641506053815],
                [-105.56226433775963, 36.07584727804726],
                [-105.53827534157463, 36.14367977962539],
                [-106.11191385205835, 36.13972769324406],
            ]
        ],
        [
            [
                [-106.11191385205835, 36.13972769324406],
                [-105.53527335203748, 36.14369323431527],
                [-105.56579262097148, 36.05653228802631],
                [-105.68980719734964, 35.659112338538634],
                [-105.85157080324588, 35.15190642354915],
                [-106.09828205302668, 35.1499211736146],
                [-106.11191385205835, 36.13972769324406],
            ]
        ],
    ],
    "coverage_epsg_candidate": ["4326", "4326"],
    "asset_epsg_candidate": [32613, 32613],
    "collection_id_benchmark": ["sentinel-2-l2a", "sentinel-2-l2a"],
    "item_id_benchmark": ["S2A_13SDV_20200403_1_L2A", "S2A_13SDV_20200403_0_L2A"],
    "item_time_benchmark": [
        Timestamp("2020-04-03 17:54:07.524000+0000", tz="utc"),
        Timestamp("2020-04-03 17:54:07.524000+0000", tz="utc"),
    ],
    "create_time_benchmark": ["2023-10-08T00:32:51.304Z", "2022-11-06T07:21:36.990Z"],
    "map_id_benchmark": [
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_1_L2A/AOT.tif",
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_0_L2A/AOT.tif",
    ],
    "map_name_benchmark": ["aot", "aot"],
    "coverage_geometry_type_benchmark": ["Polygon", "Polygon"],
    "coverage_geometry_coords_benchmark": [
        [
            [
                [-106.11191385205835, 36.13972769324406],
                [-106.09828205302668, 35.1499211736146],
                [-104.89285176524281, 35.154851672138626],
                [-104.89152152018616, 36.14484027029347],
                [-106.11191385205835, 36.13972769324406],
            ]
        ],
        [
            [
                [-106.11191385205835, 36.13972769324406],
                [-104.89152152018616, 36.14484027029347],
                [-104.89285176524281, 35.154851672138626],
                [-106.09828205302668, 35.1499211736146],
                [-106.11191385205835, 36.13972769324406],
            ]
        ],
    ],
    "coverage_epsg_benchmark": ["4326", "4326"],
    "asset_epsg_benchmark": [32613, 32613],
    "band": ["1", "1"],
    "coefficient_of_determination": [-1.449866533279419, -0.46196842193603516],
    "mean_absolute_error": [25.393386840820312, 11.947012901306152],
    "mean_absolute_percentage_error": [0.2044084370136261, 0.15074075758457184],
}


def case_stac_catalog_comparison_success():
    return url, collection, times, bbox, assets, pd.DataFrame(expected_stac_df)


bad_url = "https://google.com"
bad_times = ["2018-04-01", "1940-04-03", "2020-04-01"]
bad_assets = ["aot", "aot", "surface_water"]
exceptions = [JSONDecodeError, ValueError, ValueError]


@parametrize(
    "url, collection, time, bbox, assets, exception",
    list(
        zip(
            [bad_url, url, url],
            [collection] * 3,
            bad_times,
            [bbox] * 3,
            bad_assets,
            exceptions,
        )
    ),
)
def case_stac_catalog_comparison_fail(url, collection, time, bbox, assets, exception):
    return url, collection, time, bbox, assets, exception
