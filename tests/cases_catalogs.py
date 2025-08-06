"""
Test cases for catalogs module.
"""

# __all__ = ['*']

from pytest_cases import parametrize

import pandas as pd
from dask import dataframe as dd
import xarray as xr
import numpy as np

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
                f"{TEST_DATA_DIR}/candidate_continuous_0.tif",
                f"{TEST_DATA_DIR}/candidate_continuous_1.tif",
                f"{TEST_DATA_DIR}/candidate_continuous_1.tif",
            ],
            "compare_id": ["compare1", "compare2", "compare2"],
            "map_id_benchmark": [
                f"{TEST_DATA_DIR}/benchmark_continuous_0.tif",
                f"{TEST_DATA_DIR}/benchmark_continuous_1.tif",
                f"{TEST_DATA_DIR}/benchmark_continuous_1.tif",
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
dask = [False, False] * 2

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

expected_stac_df = [
    f"{TEST_DATA_DIR}/expected_catalog_df_no_filter.pkl",
    f"{TEST_DATA_DIR}/expected_catalog_df_allow_list.pkl",
    f"{TEST_DATA_DIR}/expected_catalog_df_block_list.pkl",
]

allow_list = [None, ["map_id", "compare_id"], None]
block_list = [None, None, ["properties.mgrs:utm_zone"]]


@parametrize(
    "url, collection, times, bbox, assets, allow_list, block_list, expected_catalog_df",
    list(
        zip(
            [url] * 3,
            [collection] * 3,
            [times] * 3,
            [bbox] * 3,
            [assets] * 3,
            allow_list,
            block_list,
            expected_stac_df,
        )
    ),
)
def case_stac_catalog_comparison_success(
    url, collection, times, bbox, assets, allow_list, block_list, expected_catalog_df
):
    return (
        url,
        collection,
        times,
        bbox,
        assets,
        allow_list,
        block_list,
        pd.read_pickle(expected_catalog_df),
    )


bad_times = ["2020-04-01"]
bad_assets = [["surface_water"], None, None, None]
exceptions = [ValueError, ValueError, KeyError, KeyError]
bad_allow_list = [None, ["arb"], ["arb"], None]
bad_block_list = [None, ["arb"], None, ["arb"]]


@parametrize(
    "url, collection, time, bbox, assets, allow_list, block_list, exception",
    list(
        zip(
            [url] * len(bad_assets),
            [collection] * len(bad_assets),
            bad_times * len(bad_assets),
            [bbox] * len(bad_assets),
            bad_assets,
            bad_allow_list,
            bad_block_list,
            exceptions,
        )
    ),
)
def case_stac_catalog_comparison_fail(
    url, collection, time, bbox, assets, allow_list, block_list, exception
):
    return url, collection, time, bbox, assets, allow_list, block_list, exception
