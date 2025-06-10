"""
Tests for the catalog module
"""

from pytest_cases import parametrize_with_cases
from pytest import raises

import pandas as pd
import dask.dataframe as dd
import rioxarray as rxr
import xarray as xr
import pystac_client

from tests.conftest import _attributes_to_string
from gval.catalogs.catalogs import catalog_compare
from gval.utils.loading_datasets import stac_to_df, _parse_string_attributes


@parametrize_with_cases(
    "candidate_catalog, benchmark_catalog, on, map_ids, how, compare_type, compare_kwargs, open_kwargs, agreement_map_field, expected, expected_agreement_map",
    glob="compare_catalogs",
)
def test_compare_catalogs(
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
    tmp_path,
):
    """Tests for the compare_catalogs function"""

    def prepend_tmp_path(df, field):
        """Prepend tmp_path to field in df"""
        if (field is not None) & (field in df.columns):

            def tmp_path_apply(row):
                return tmp_path / row[field]

            dask_meta = (
                {"meta": ("output", "O")} if isinstance(df, dd.DataFrame) else {}
            )
            df[field] = df.apply(tmp_path_apply, axis=1, **dask_meta)
        return df

    # apply tmp_path to agreement_map_field in all catalogs
    candidate_catalog = prepend_tmp_path(candidate_catalog, agreement_map_field)
    benchmark_catalog = prepend_tmp_path(benchmark_catalog, agreement_map_field)
    expected = prepend_tmp_path(expected, agreement_map_field)

    agreement_catalog = catalog_compare(
        candidate_catalog,
        benchmark_catalog,
        on=on,
        map_ids=map_ids,
        how=how,
        compare_type=compare_type,
        compare_kwargs=compare_kwargs,
        open_kwargs=open_kwargs,
        agreement_map_field=agreement_map_field,
    )

    if isinstance(agreement_catalog, dd.DataFrame):
        agreement_catalog = agreement_catalog.compute().reset_index(drop=True)

    # check that the columns are the same without order
    pd.testing.assert_index_equal(
        agreement_catalog.columns, expected.columns, check_order=False
    )

    # check that the columns are the same with order. if not, reorder
    try:
        pd.testing.assert_index_equal(
            agreement_catalog.columns, expected.columns, check_order=True
        )
    except AssertionError:
        # make the same order of columns
        agreement_catalog = agreement_catalog[expected.columns]
        pd.testing.assert_index_equal(
            agreement_catalog.columns, expected.columns, check_order=True
        )

    # check dtypes. if not the same, recast
    agreement_catalog = agreement_catalog.astype(expected.dtypes)
    pd.testing.assert_series_equal(agreement_catalog.dtypes, expected.dtypes)
    agreement_catalog["agreement_maps"] = agreement_catalog["agreement_maps"].astype(
        str
    )
    expected["agreement_maps"] = expected["agreement_maps"].astype(str)

    pd.testing.assert_frame_equal(
        agreement_catalog, expected, check_dtype=False, check_like=True
    )

    # load agreement maps and check metadata
    if agreement_map_field is not None:
        # load agreement maps with apply and check that they are the same

        def load_agreement_and_check(row, counter=[0]):
            """Load agreement map and check that it is the same"""
            agreement_map = rxr.open_rasterio(row[agreement_map_field], **open_kwargs)

            if expected_agreement_map[counter[0]] is not None:
                expected_agreement_map_xr = rxr.open_rasterio(
                    expected_agreement_map[counter[0]], **open_kwargs
                )

                xr.testing.assert_identical(
                    _attributes_to_string(_parse_string_attributes(agreement_map)),
                    _attributes_to_string(expected_agreement_map_xr),
                )

            # increment counter
            counter[0] += 1

        _ = agreement_catalog.apply(load_agreement_and_check, axis=1)


@parametrize_with_cases(
    "candidate_catalog, benchmark_catalog, on, map_ids, how, compare_type, compare_kwargs, open_kwargs, agreement_map_field, expected, expected_agreement_map",
    glob="compare_catalogs_no_kwargs",
)
def test_compare_catalogs_no_kwargs(
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
    tmp_path,
):
    """Tests for the _compare_catalogs function"""

    def prepend_tmp_path(df, field):
        """Prepend tmp_path to field in df"""
        if (field is not None) & (field in df.columns):

            def tmp_path_apply(row):
                return tmp_path / row[field]

            dask_meta = (
                {"meta": ("output", "O")} if isinstance(df, dd.DataFrame) else {}
            )
            df[field] = df.apply(tmp_path_apply, axis=1, **dask_meta)
        return df

    # apply tmp_path to agreement_map_field in all catalogs
    candidate_catalog = prepend_tmp_path(candidate_catalog, agreement_map_field)
    benchmark_catalog = prepend_tmp_path(benchmark_catalog, agreement_map_field)
    expected = prepend_tmp_path(expected, agreement_map_field)

    agreement_catalog = catalog_compare(
        candidate_catalog,
        benchmark_catalog,
        on=on,
        map_ids=map_ids,
        how=how,
        compare_type=compare_type,
        compare_kwargs=compare_kwargs,
        open_kwargs=open_kwargs,
        agreement_map_field=agreement_map_field,
    )

    assert isinstance(agreement_catalog, dd.DataFrame) or isinstance(
        agreement_catalog, pd.DataFrame
    )

    # load agreement maps and check metadata
    if agreement_map_field is not None:
        # load agreement maps with apply and check that they are the same

        def load_agreement_and_check(row, counter=[0]):
            """Load agreement map and check that it is the same"""
            agreement_map = rxr.open_rasterio(row[agreement_map_field], **open_kwargs)

            if expected_agreement_map[counter[0]] is not None:
                expected_agreement_map_xr = rxr.open_rasterio(
                    expected_agreement_map[counter[0]], **open_kwargs
                )
                assert isinstance(agreement_map, (xr.DataArray, xr.Dataset))
                assert isinstance(expected_agreement_map_xr, (xr.DataArray, xr.Dataset))


@parametrize_with_cases(
    "candidate_catalog, benchmark_catalog, on, map_ids, how, compare_type, compare_kwargs, open_kwargs, agreement_map_field, expected_exception",
    glob="compare_catalogs_fail",
)
def test_compare_catalogs_fail(
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
    tmp_path,
):
    """Tests for the compare_catalogs function"""

    with raises(expected_exception):
        catalog_compare(
            candidate_catalog,
            benchmark_catalog,
            on=on,
            map_ids=map_ids,
            how=how,
            compare_type=compare_type,
            compare_kwargs=compare_kwargs,
            open_kwargs=open_kwargs,
            agreement_map_field=agreement_map_field,
        )


# This test needs to be re-evaluated
# @parametrize_with_cases(
#     "url, collection, times, bbox, assets, allow_list, block_list, expected_catalog_df",
#     glob="stac_catalog_comparison_success",
# )
# def test_stac_catalog_comparison_success(
#     url, collection, times, bbox, assets, allow_list, block_list, expected_catalog_df
# ):
#     catalog = pystac_client.Client.open(url)
#
#     candidate_items = catalog.search(
#         datetime=times[0],
#         collections=[collection],
#         bbox=bbox,
#     ).item_collection()
#
#     candidate_catalog = stac_to_df(
#         stac_items=candidate_items,
#         assets=assets,
#         attribute_allow_list=allow_list,
#         attribute_block_list=block_list,
#     )
#
#     benchmark_items = catalog.search(
#         datetime=times[1],
#         collections=[collection],
#         bbox=bbox,
#     ).item_collection()
#
#     benchmark_catalog = stac_to_df(
#         stac_items=benchmark_items,
#         assets=assets,
#         attribute_allow_list=allow_list,
#         attribute_block_list=block_list,
#     )
#
#     arguments = {
#         "candidate_catalog": candidate_catalog,
#         "benchmark_catalog": benchmark_catalog,
#         "on": "compare_id",
#         "map_ids": "map_id",
#         "how": "inner",
#         "compare_type": "continuous",
#         "compare_kwargs": {
#             "metrics": (
#                 "coefficient_of_determination",
#                 "mean_absolute_error",
#                 "mean_absolute_percentage_error",
#             ),
#             "encode_nodata": True,
#             "nodata": -9999,
#         },
#         "open_kwargs": {"mask_and_scale": True, "masked": True},
#     }
#
#     stac_clog = catalog_compare(**arguments)
#
#     pd.testing.assert_frame_equal(
#         stac_clog,
#         expected_catalog_df,
#         check_dtype=True,
#         check_index_type=False,
#         check_like=True,
#     ), "Computed catalog did not match the expected catalog df"


@parametrize_with_cases(
    "url, collection, time, bbox, assets, allow_list, block_list, exception",
    glob="stac_catalog_comparison_fail",
)
def test_stac_catalog_comparison_fail(
    url, collection, time, bbox, assets, allow_list, block_list, exception
):
    with raises(exception):
        catalog = pystac_client.Client.open(url)

        candidate_items = catalog.search(
            datetime=time,
            collections=[collection],
            bbox=bbox,
        ).item_collection()

        _ = stac_to_df(
            stac_items=candidate_items,
            assets=assets,
            attribute_allow_list=allow_list,
            attribute_block_list=block_list,
        )
