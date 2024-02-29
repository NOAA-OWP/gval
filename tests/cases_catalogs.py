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

expected_stac_df = [
    {
        "created_candidate": ["2023-10-07T12:09:07.273Z", "2022-11-06T10:14:16.681Z"],
        "platform_candidate": ["sentinel-2b", "sentinel-2b"],
        "constellation_candidate": ["sentinel-2", "sentinel-2"],
        "eo:cloud_cover_candidate": [26.25798, 26.031335],
        "proj:epsg_candidate": [32613, 32613],
        "mgrs:utm_zone_candidate": [13, 13],
        "mgrs:latitude_band_candidate": ["S", "S"],
        "mgrs:grid_square_candidate": ["DV", "DV"],
        "grid:code_candidate": ["MGRS-13SDV", "MGRS-13SDV"],
        "view:sun_azimuth_candidate": [151.651828166802, 151.650282024803],
        "view:sun_elevation_candidate": [56.1648077338387, 56.164320019392],
        "s2:degraded_msi_data_percentage_candidate": [0.0165, 0.0],
        "s2:nodata_pixel_percentage_candidate": [66.295946, 66.037267],
        "s2:saturated_defective_pixel_percentage_candidate": [0, 0],
        "s2:dark_features_percentage_candidate": [2.557362, 4.251513],
        "s2:cloud_shadow_percentage_candidate": [0.709105, 1.063333],
        "s2:vegetation_percentage_candidate": [6.046846, 6.331863],
        "s2:not_vegetated_percentage_candidate": [55.535203, 53.858513],
        "s2:water_percentage_candidate": [0.019934, 0.045153],
        "s2:unclassified_percentage_candidate": [5.82255, 5.403911],
        "s2:medium_proba_clouds_percentage_candidate": [5.997636, 6.010647],
        "s2:high_proba_clouds_percentage_candidate": [20.260343, 20.020688],
        "s2:thin_cirrus_percentage_candidate": [0, 0],
        "s2:snow_ice_percentage_candidate": [3.051021, 3.01438],
        "s2:product_type_candidate": ["S2MSI2A", "S2MSI2A"],
        "s2:processing_baseline_candidate": ["05.00", "02.14"],
        "s2:product_uri_candidate": [
            "S2B_MSIL2A_20200401T174909_N0500_R141_T13SDV_20230624T053545.SAFE",
            "S2B_MSIL2A_20200401T174909_N0214_R141_T13SDV_20200401T220155.SAFE",
        ],
        "s2:generation_time_candidate": [
            "2023-06-24T05:35:45.000000Z",
            "2020-04-01T22:01:55.000000Z",
        ],
        "s2:datatake_id_candidate": [
            "GS2B_20200401T174909_016040_N05.00",
            "GS2B_20200401T174909_016040_N02.14",
        ],
        "s2:datatake_type_candidate": ["INS-NOBS", "INS-NOBS"],
        "s2:datastrip_id_candidate": [
            "S2B_OPER_MSI_L2A_DS_S2RP_20230624T053545_S20200401T175716_N05.00",
            "S2B_OPER_MSI_L2A_DS_EPAE_20200401T220155_S20200401T175716_N02.14",
        ],
        "s2:granule_id_candidate": [
            "S2B_OPER_MSI_L2A_TL_S2RP_20230624T053545_A016040_T13SDV_N05.00",
            "S2B_OPER_MSI_L2A_TL_EPAE_20200401T220155_A016040_T13SDV_N02.14",
        ],
        "s2:reflectance_conversion_factor_candidate": [
            1.00356283682453,
            1.00356283682453,
        ],
        "datetime_candidate": [
            "2020-04-01T18:04:04.327000Z",
            "2020-04-01T18:04:04.327000Z",
        ],
        "s2:sequence_candidate": ["1", "0"],
        "earthsearch:s3_path_candidate": [
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_1_L2A",
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_0_L2A",
        ],
        "earthsearch:payload_id_candidate": [
            "roda-sentinel2/workflow-sentinel2-to-stac/71c0289236fa3e831ac2f8c860df8cae",
            "roda-sentinel2/workflow-sentinel2-to-stac/afb43c585d466972865ed5139ba35520",
        ],
        "earthsearch:boa_offset_applied_candidate": [True, False],
        "sentinel2-to-stac_candidate": ["0.1.1", "0.1.0"],
        "updated_candidate": ["2023-10-07T12:09:07.273Z", "2022-11-06T10:14:16.681Z"],
        "bbox_candidate": [
            "MULTIPOINT (-106.11191385205835 35.150822790058335, -105.53827534157463 36.14367977962539)",
            "MULTIPOINT (-106.11191385205835 35.1499211736146, -105.53527335203748 36.14369323431527)",
        ],
        "geometry_candidate": [
            "POLYGON ((-106.11191385205835 36.13972769324406, -106.0982941692468 35.150822790058335, -105.85393882465051 35.15385918076215, -105.68102037327243 35.69893282033011, -105.59450557216445 35.97641506053815, -105.56226433775963 36.07584727804726, -105.53827534157463 36.14367977962539, -106.11191385205835 36.13972769324406))",
            "POLYGON ((-106.11191385205835 36.13972769324406, -105.53527335203748 36.14369323431527, -105.56579262097148 36.05653228802631, -105.68980719734964 35.659112338538634, -105.85157080324588 35.15190642354915, -106.09828205302668 35.1499211736146, -106.11191385205835 36.13972769324406))",
        ],
        "roles_candidate": ["['data', 'reflectance']", "['data', 'reflectance']"],
        "compare_id": [1, 2],
        "map_id_candidate": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_0_L2A/AOT.tif",
        ],
        "asset_candidate": ["aot", "aot"],
        "href_candidate": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_0_L2A/AOT.tif",
        ],
        "type_candidate": [
            "image/tiff; application=geotiff; profile=cloud-optimized",
            "image/tiff; application=geotiff; profile=cloud-optimized",
        ],
        "title_candidate": [
            "Aerosol optical thickness (AOT)",
            "Aerosol optical thickness (AOT)",
        ],
        "proj:shape_candidate": ["[5490, 5490]", "[5490, 5490]"],
        "proj:transform_candidate": [
            "[20, 0, 399960, 0, -20, 4000020]",
            "[20, 0, 399960, 0, -20, 4000020]",
        ],
        "raster:bands_candidate": [
            "[{'nodata': 0, 'data_type': 'uint16', 'bits_per_sample': 15, 'spatial_resolution': 20, 'scale': 0.001, 'offset': 0}]",
            "[{'nodata': 0, 'data_type': 'uint16', 'bits_per_sample': 15, 'spatial_resolution': 20, 'scale': 0.001, 'offset': 0}]",
        ],
        "eo:bands_candidate": ["N/a", "N/a"],
        "gsd_candidate": ["N/a", "N/a"],
        "created_benchmark": ["2023-10-08T00:32:51.304Z", "2022-11-06T07:21:36.990Z"],
        "platform_benchmark": ["sentinel-2a", "sentinel-2a"],
        "constellation_benchmark": ["sentinel-2", "sentinel-2"],
        "eo:cloud_cover_benchmark": [0.394644, 0.946059],
        "proj:epsg_benchmark": [32613, 32613],
        "mgrs:utm_zone_benchmark": [13, 13],
        "mgrs:latitude_band_benchmark": ["S", "S"],
        "mgrs:grid_square_benchmark": ["DV", "DV"],
        "grid:code_benchmark": ["MGRS-13SDV", "MGRS-13SDV"],
        "view:sun_azimuth_benchmark": [147.277474939279, 147.275985208519],
        "view:sun_elevation_benchmark": [55.8900570304106, 55.8895151168133],
        "s2:degraded_msi_data_percentage_benchmark": [0, 0],
        "s2:nodata_pixel_percentage_benchmark": [0, 0],
        "s2:saturated_defective_pixel_percentage_benchmark": [0, 0],
        "s2:dark_features_percentage_benchmark": [0.619855, 0.411445],
        "s2:cloud_shadow_percentage_benchmark": [0.074044, 0.143752],
        "s2:vegetation_percentage_benchmark": [13.550997, 13.711172],
        "s2:not_vegetated_percentage_benchmark": [79.290646, 77.119505],
        "s2:water_percentage_benchmark": [0.095783, 0.093679],
        "s2:unclassified_percentage_benchmark": [0.236433, 2.878225],
        "s2:medium_proba_clouds_percentage_benchmark": [0.262985, 0.760678],
        "s2:high_proba_clouds_percentage_benchmark": [0.131658, 0.185381],
        "s2:thin_cirrus_percentage_benchmark": [0, 0],
        "s2:snow_ice_percentage_benchmark": [5.737595, 4.696162],
        "s2:product_type_benchmark": ["S2MSI2A", "S2MSI2A"],
        "s2:processing_baseline_benchmark": ["05.00", "02.14"],
        "s2:product_uri_benchmark": [
            "S2A_MSIL2A_20200403T173901_N0500_R098_T13SDV_20230510T053445.SAFE",
            "S2A_MSIL2A_20200403T173901_N0214_R098_T13SDV_20200403T220105.SAFE",
        ],
        "s2:generation_time_benchmark": [
            "2023-05-10T05:34:45.000000Z",
            "2020-04-03T22:01:05.000000Z",
        ],
        "s2:datatake_id_benchmark": [
            "GS2A_20200403T173901_024977_N05.00",
            "GS2A_20200403T173901_024977_N02.14",
        ],
        "s2:datatake_type_benchmark": ["INS-NOBS", "INS-NOBS"],
        "s2:datastrip_id_benchmark": [
            "S2A_OPER_MSI_L2A_DS_S2RP_20230510T053445_S20200403T174815_N05.00",
            "S2A_OPER_MSI_L2A_DS_SGS__20200403T220105_S20200403T174815_N02.14",
        ],
        "s2:granule_id_benchmark": [
            "S2A_OPER_MSI_L2A_TL_S2RP_20230510T053445_A024977_T13SDV_N05.00",
            "S2A_OPER_MSI_L2A_TL_SGS__20200403T220105_A024977_T13SDV_N02.14",
        ],
        "s2:reflectance_conversion_factor_benchmark": [
            1.00241535908783,
            1.00241535908783,
        ],
        "datetime_benchmark": [
            "2020-04-03T17:54:07.524000Z",
            "2020-04-03T17:54:07.524000Z",
        ],
        "s2:sequence_benchmark": ["1", "0"],
        "earthsearch:s3_path_benchmark": [
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_1_L2A",
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_0_L2A",
        ],
        "earthsearch:payload_id_benchmark": [
            "roda-sentinel2/workflow-sentinel2-to-stac/5c5486e239b6fb66c09401d57834e542",
            "roda-sentinel2/workflow-sentinel2-to-stac/00bfe5eff22b9aae0b5adfae696a11fa",
        ],
        "earthsearch:boa_offset_applied_benchmark": [True, False],
        "sentinel2-to-stac_benchmark": ["0.1.1", "0.1.0"],
        "updated_benchmark": ["2023-10-08T00:32:51.304Z", "2022-11-06T07:21:36.990Z"],
        "bbox_benchmark": [
            "MULTIPOINT (-106.11191385205835 35.1499211736146, -104.89152152018616 36.14484027029347)",
            "MULTIPOINT (-106.11191385205835 35.1499211736146, -104.89152152018616 36.14484027029347)",
        ],
        "geometry_benchmark": [
            "POLYGON ((-106.11191385205835 36.13972769324406, -106.09828205302668 35.1499211736146, -104.89285176524281 35.154851672138626, -104.89152152018616 36.14484027029347, -106.11191385205835 36.13972769324406))",
            "POLYGON ((-106.11191385205835 36.13972769324406, -104.89152152018616 36.14484027029347, -104.89285176524281 35.154851672138626, -106.09828205302668 35.1499211736146, -106.11191385205835 36.13972769324406))",
        ],
        "roles_benchmark": ["['data', 'reflectance']", "['data', 'reflectance']"],
        "map_id_benchmark": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_0_L2A/AOT.tif",
        ],
        "asset_benchmark": ["aot", "aot"],
        "href_benchmark": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_0_L2A/AOT.tif",
        ],
        "type_benchmark": [
            "image/tiff; application=geotiff; profile=cloud-optimized",
            "image/tiff; application=geotiff; profile=cloud-optimized",
        ],
        "title_benchmark": [
            "Aerosol optical thickness (AOT)",
            "Aerosol optical thickness (AOT)",
        ],
        "proj:shape_benchmark": ["[5490, 5490]", "[5490, 5490]"],
        "proj:transform_benchmark": [
            "[20, 0, 399960, 0, -20, 4000020]",
            "[20, 0, 399960, 0, -20, 4000020]",
        ],
        "raster:bands_benchmark": [
            "[{'nodata': 0, 'data_type': 'uint16', 'bits_per_sample': 15, 'spatial_resolution': 20, 'scale': 0.001, 'offset': 0}]",
            "[{'nodata': 0, 'data_type': 'uint16', 'bits_per_sample': 15, 'spatial_resolution': 20, 'scale': 0.001, 'offset': 0}]",
        ],
        "eo:bands_benchmark": ["N/a", "N/a"],
        "gsd_benchmark": ["N/a", "N/a"],
        "band": ["1", "1"],
        "coefficient_of_determination": [-1.449866533279419, -0.46196842193603516],
        "mean_absolute_error": [25.393386840820312, 11.947012901306152],
        "mean_absolute_percentage_error": [0.2044084370136261, 0.15074075758457184],
    },
    {
        "map_id_candidate": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_0_L2A/AOT.tif",
        ],
        "compare_id": [1, 2],
        "map_id_benchmark": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_0_L2A/AOT.tif",
        ],
        "band": ["1", "1"],
        "coefficient_of_determination": [-1.449866533279419, -0.46196842193603516],
        "mean_absolute_error": [25.393386840820312, 11.947012901306152],
        "mean_absolute_percentage_error": [0.2044084370136261, 0.15074075758457184],
    },
    {
        "created_candidate": ["2023-10-07T12:09:07.273Z", "2022-11-06T10:14:16.681Z"],
        "platform_candidate": ["sentinel-2b", "sentinel-2b"],
        "constellation_candidate": ["sentinel-2", "sentinel-2"],
        "proj:epsg_candidate": [32613, 32613],
        "mgrs:utm_zone_candidate": [13, 13],
        "mgrs:latitude_band_candidate": ["S", "S"],
        "mgrs:grid_square_candidate": ["DV", "DV"],
        "grid:code_candidate": ["MGRS-13SDV", "MGRS-13SDV"],
        "view:sun_azimuth_candidate": [151.651828166802, 151.650282024803],
        "view:sun_elevation_candidate": [56.1648077338387, 56.164320019392],
        "s2:degraded_msi_data_percentage_candidate": [0.0165, 0.0],
        "s2:nodata_pixel_percentage_candidate": [66.295946, 66.037267],
        "s2:saturated_defective_pixel_percentage_candidate": [0, 0],
        "s2:dark_features_percentage_candidate": [2.557362, 4.251513],
        "s2:cloud_shadow_percentage_candidate": [0.709105, 1.063333],
        "s2:vegetation_percentage_candidate": [6.046846, 6.331863],
        "s2:not_vegetated_percentage_candidate": [55.535203, 53.858513],
        "s2:water_percentage_candidate": [0.019934, 0.045153],
        "s2:unclassified_percentage_candidate": [5.82255, 5.403911],
        "s2:medium_proba_clouds_percentage_candidate": [5.997636, 6.010647],
        "s2:high_proba_clouds_percentage_candidate": [20.260343, 20.020688],
        "s2:thin_cirrus_percentage_candidate": [0, 0],
        "s2:snow_ice_percentage_candidate": [3.051021, 3.01438],
        "s2:product_type_candidate": ["S2MSI2A", "S2MSI2A"],
        "s2:processing_baseline_candidate": ["05.00", "02.14"],
        "s2:product_uri_candidate": [
            "S2B_MSIL2A_20200401T174909_N0500_R141_T13SDV_20230624T053545.SAFE",
            "S2B_MSIL2A_20200401T174909_N0214_R141_T13SDV_20200401T220155.SAFE",
        ],
        "s2:generation_time_candidate": [
            "2023-06-24T05:35:45.000000Z",
            "2020-04-01T22:01:55.000000Z",
        ],
        "s2:datatake_id_candidate": [
            "GS2B_20200401T174909_016040_N05.00",
            "GS2B_20200401T174909_016040_N02.14",
        ],
        "s2:datatake_type_candidate": ["INS-NOBS", "INS-NOBS"],
        "s2:datastrip_id_candidate": [
            "S2B_OPER_MSI_L2A_DS_S2RP_20230624T053545_S20200401T175716_N05.00",
            "S2B_OPER_MSI_L2A_DS_EPAE_20200401T220155_S20200401T175716_N02.14",
        ],
        "s2:granule_id_candidate": [
            "S2B_OPER_MSI_L2A_TL_S2RP_20230624T053545_A016040_T13SDV_N05.00",
            "S2B_OPER_MSI_L2A_TL_EPAE_20200401T220155_A016040_T13SDV_N02.14",
        ],
        "s2:reflectance_conversion_factor_candidate": [
            1.00356283682453,
            1.00356283682453,
        ],
        "datetime_candidate": [
            "2020-04-01T18:04:04.327000Z",
            "2020-04-01T18:04:04.327000Z",
        ],
        "s2:sequence_candidate": ["1", "0"],
        "earthsearch:s3_path_candidate": [
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_1_L2A",
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_0_L2A",
        ],
        "earthsearch:payload_id_candidate": [
            "roda-sentinel2/workflow-sentinel2-to-stac/71c0289236fa3e831ac2f8c860df8cae",
            "roda-sentinel2/workflow-sentinel2-to-stac/afb43c585d466972865ed5139ba35520",
        ],
        "earthsearch:boa_offset_applied_candidate": [True, False],
        "sentinel2-to-stac_candidate": ["0.1.1", "0.1.0"],
        "updated_candidate": ["2023-10-07T12:09:07.273Z", "2022-11-06T10:14:16.681Z"],
        "bbox_candidate": [
            "MULTIPOINT (-106.11191385205835 35.150822790058335, -105.53827534157463 36.14367977962539)",
            "MULTIPOINT (-106.11191385205835 35.1499211736146, -105.53527335203748 36.14369323431527)",
        ],
        "geometry_candidate": [
            "POLYGON ((-106.11191385205835 36.13972769324406, -106.0982941692468 35.150822790058335, -105.85393882465051 35.15385918076215, -105.68102037327243 35.69893282033011, -105.59450557216445 35.97641506053815, -105.56226433775963 36.07584727804726, -105.53827534157463 36.14367977962539, -106.11191385205835 36.13972769324406))",
            "POLYGON ((-106.11191385205835 36.13972769324406, -105.53527335203748 36.14369323431527, -105.56579262097148 36.05653228802631, -105.68980719734964 35.659112338538634, -105.85157080324588 35.15190642354915, -106.09828205302668 35.1499211736146, -106.11191385205835 36.13972769324406))",
        ],
        "roles_candidate": ["['data', 'reflectance']", "['data', 'reflectance']"],
        "compare_id": [1, 2],
        "map_id_candidate": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_0_L2A/AOT.tif",
        ],
        "asset_candidate": ["aot", "aot"],
        "href_candidate": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2B_13SDV_20200401_0_L2A/AOT.tif",
        ],
        "type_candidate": [
            "image/tiff; application=geotiff; profile=cloud-optimized",
            "image/tiff; application=geotiff; profile=cloud-optimized",
        ],
        "title_candidate": [
            "Aerosol optical thickness (AOT)",
            "Aerosol optical thickness (AOT)",
        ],
        "proj:shape_candidate": ["[5490, 5490]", "[5490, 5490]"],
        "proj:transform_candidate": [
            "[20, 0, 399960, 0, -20, 4000020]",
            "[20, 0, 399960, 0, -20, 4000020]",
        ],
        "raster:bands_candidate": [
            "[{'nodata': 0, 'data_type': 'uint16', 'bits_per_sample': 15, 'spatial_resolution': 20, 'scale': 0.001, 'offset': 0}]",
            "[{'nodata': 0, 'data_type': 'uint16', 'bits_per_sample': 15, 'spatial_resolution': 20, 'scale': 0.001, 'offset': 0}]",
        ],
        "eo:bands_candidate": ["N/a", "N/a"],
        "gsd_candidate": ["N/a", "N/a"],
        "created_benchmark": ["2023-10-08T00:32:51.304Z", "2022-11-06T07:21:36.990Z"],
        "platform_benchmark": ["sentinel-2a", "sentinel-2a"],
        "constellation_benchmark": ["sentinel-2", "sentinel-2"],
        "proj:epsg_benchmark": [32613, 32613],
        "mgrs:utm_zone_benchmark": [13, 13],
        "mgrs:latitude_band_benchmark": ["S", "S"],
        "mgrs:grid_square_benchmark": ["DV", "DV"],
        "grid:code_benchmark": ["MGRS-13SDV", "MGRS-13SDV"],
        "view:sun_azimuth_benchmark": [147.277474939279, 147.275985208519],
        "view:sun_elevation_benchmark": [55.8900570304106, 55.8895151168133],
        "s2:degraded_msi_data_percentage_benchmark": [0, 0],
        "s2:nodata_pixel_percentage_benchmark": [0, 0],
        "s2:saturated_defective_pixel_percentage_benchmark": [0, 0],
        "s2:dark_features_percentage_benchmark": [0.619855, 0.411445],
        "s2:cloud_shadow_percentage_benchmark": [0.074044, 0.143752],
        "s2:vegetation_percentage_benchmark": [13.550997, 13.711172],
        "s2:not_vegetated_percentage_benchmark": [79.290646, 77.119505],
        "s2:water_percentage_benchmark": [0.095783, 0.093679],
        "s2:unclassified_percentage_benchmark": [0.236433, 2.878225],
        "s2:medium_proba_clouds_percentage_benchmark": [0.262985, 0.760678],
        "s2:high_proba_clouds_percentage_benchmark": [0.131658, 0.185381],
        "s2:thin_cirrus_percentage_benchmark": [0, 0],
        "s2:snow_ice_percentage_benchmark": [5.737595, 4.696162],
        "s2:product_type_benchmark": ["S2MSI2A", "S2MSI2A"],
        "s2:processing_baseline_benchmark": ["05.00", "02.14"],
        "s2:product_uri_benchmark": [
            "S2A_MSIL2A_20200403T173901_N0500_R098_T13SDV_20230510T053445.SAFE",
            "S2A_MSIL2A_20200403T173901_N0214_R098_T13SDV_20200403T220105.SAFE",
        ],
        "s2:generation_time_benchmark": [
            "2023-05-10T05:34:45.000000Z",
            "2020-04-03T22:01:05.000000Z",
        ],
        "s2:datatake_id_benchmark": [
            "GS2A_20200403T173901_024977_N05.00",
            "GS2A_20200403T173901_024977_N02.14",
        ],
        "s2:datatake_type_benchmark": ["INS-NOBS", "INS-NOBS"],
        "s2:datastrip_id_benchmark": [
            "S2A_OPER_MSI_L2A_DS_S2RP_20230510T053445_S20200403T174815_N05.00",
            "S2A_OPER_MSI_L2A_DS_SGS__20200403T220105_S20200403T174815_N02.14",
        ],
        "s2:granule_id_benchmark": [
            "S2A_OPER_MSI_L2A_TL_S2RP_20230510T053445_A024977_T13SDV_N05.00",
            "S2A_OPER_MSI_L2A_TL_SGS__20200403T220105_A024977_T13SDV_N02.14",
        ],
        "s2:reflectance_conversion_factor_benchmark": [
            1.00241535908783,
            1.00241535908783,
        ],
        "datetime_benchmark": [
            "2020-04-03T17:54:07.524000Z",
            "2020-04-03T17:54:07.524000Z",
        ],
        "s2:sequence_benchmark": ["1", "0"],
        "earthsearch:s3_path_benchmark": [
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_1_L2A",
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_0_L2A",
        ],
        "earthsearch:payload_id_benchmark": [
            "roda-sentinel2/workflow-sentinel2-to-stac/5c5486e239b6fb66c09401d57834e542",
            "roda-sentinel2/workflow-sentinel2-to-stac/00bfe5eff22b9aae0b5adfae696a11fa",
        ],
        "earthsearch:boa_offset_applied_benchmark": [True, False],
        "sentinel2-to-stac_benchmark": ["0.1.1", "0.1.0"],
        "updated_benchmark": ["2023-10-08T00:32:51.304Z", "2022-11-06T07:21:36.990Z"],
        "bbox_benchmark": [
            "MULTIPOINT (-106.11191385205835 35.1499211736146, -104.89152152018616 36.14484027029347)",
            "MULTIPOINT (-106.11191385205835 35.1499211736146, -104.89152152018616 36.14484027029347)",
        ],
        "geometry_benchmark": [
            "POLYGON ((-106.11191385205835 36.13972769324406, -106.09828205302668 35.1499211736146, -104.89285176524281 35.154851672138626, -104.89152152018616 36.14484027029347, -106.11191385205835 36.13972769324406))",
            "POLYGON ((-106.11191385205835 36.13972769324406, -104.89152152018616 36.14484027029347, -104.89285176524281 35.154851672138626, -106.09828205302668 35.1499211736146, -106.11191385205835 36.13972769324406))",
        ],
        "roles_benchmark": ["['data', 'reflectance']", "['data', 'reflectance']"],
        "map_id_benchmark": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_0_L2A/AOT.tif",
        ],
        "asset_benchmark": ["aot", "aot"],
        "href_benchmark": [
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_1_L2A/AOT.tif",
            "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/13/S/DV/2020/4/S2A_13SDV_20200403_0_L2A/AOT.tif",
        ],
        "type_benchmark": [
            "image/tiff; application=geotiff; profile=cloud-optimized",
            "image/tiff; application=geotiff; profile=cloud-optimized",
        ],
        "title_benchmark": [
            "Aerosol optical thickness (AOT)",
            "Aerosol optical thickness (AOT)",
        ],
        "proj:shape_benchmark": ["[5490, 5490]", "[5490, 5490]"],
        "proj:transform_benchmark": [
            "[20, 0, 399960, 0, -20, 4000020]",
            "[20, 0, 399960, 0, -20, 4000020]",
        ],
        "raster:bands_benchmark": [
            "[{'nodata': 0, 'data_type': 'uint16', 'bits_per_sample': 15, 'spatial_resolution': 20, 'scale': 0.001, 'offset': 0}]",
            "[{'nodata': 0, 'data_type': 'uint16', 'bits_per_sample': 15, 'spatial_resolution': 20, 'scale': 0.001, 'offset': 0}]",
        ],
        "eo:bands_benchmark": ["N/a", "N/a"],
        "gsd_benchmark": ["N/a", "N/a"],
        "band": ["1", "1"],
        "coefficient_of_determination": [-1.449866533279419, -0.46196842193603516],
        "mean_absolute_error": [25.393386840820312, 11.947012901306152],
        "mean_absolute_percentage_error": [0.2044084370136261, 0.15074075758457184],
    },
]

allow_list = [None, ["map_id", "compare_id"], None]
block_list = [None, None, ["eo:cloud_cover"]]


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
        pd.DataFrame(expected_catalog_df),
    )


bad_times = ["2020-04-01"]
bad_assets = ["surface_water", None, None]
exceptions = [ValueError, KeyError, KeyError]
bad_allow_list = [None, ["arb"], None]
bad_block_list = [None, None, ["arb"]]


@parametrize(
    "url, collection, time, bbox, assets, allow_list, block_list, exception",
    list(
        zip(
            [url] * 3,
            [collection] * 3,
            bad_times * 3,
            [bbox] * 3,
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
