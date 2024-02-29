from pytest_cases import parametrize_with_cases
from pytest import raises
import xarray as xr
import pandas as pd
import pystac_client

from gval.utils.loading_datasets import get_stac_data


@parametrize_with_cases(
    "url, collection, bbox, time, bands, time_aggregate, nodata_fill, expected_df",
    glob="stac_api_call",
)
def test_stac_api_call(
    url, collection, bbox, time, bands, time_aggregate, nodata_fill, expected_df
):
    """
    Tests call for stac API, (IDK if this data can be mocked, API calls in unit tests are dubious)
    """

    catalog = pystac_client.Client.open(url)

    candidate_items = catalog.search(
        datetime=time[0],
        collections=[collection],
        bbox=bbox,
    ).item_collection()

    candidate = get_stac_data(
        stac_items=candidate_items,
        bands=bands,
        time_aggregate=time_aggregate,
        nodata_fill=nodata_fill,
    )

    benchmark_items = catalog.search(
        datetime=time[1],
        collections=[collection],
        bbox=bbox,
    ).item_collection()

    benchmark = get_stac_data(
        stac_items=benchmark_items,
        bands=bands,
        time_aggregate=time_aggregate,
        nodata_fill=nodata_fill,
    )

    # Registered metrics from previous tests affect this comparison so metrics are explicit
    agreement, metrics = candidate.gval.continuous_compare(
        benchmark,
        metrics=[
            "coefficient_of_determination",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
            "mean_normalized_mean_absolute_error",
            "mean_normalized_root_mean_squared_error",
            "mean_percentage_error",
            "mean_signed_error",
            "mean_squared_error",
            "range_normalized_mean_absolute_error",
            "range_normalized_root_mean_squared_error",
            "root_mean_squared_error",
            "symmetric_mean_absolute_percentage_error",
        ],
    )

    bnds = [bands] if isinstance(bands, str) else bands

    assert isinstance(agreement, xr.Dataset)
    assert bnds == [
        agreement[var].attrs["original_name"] for var in agreement.data_vars
    ]

    pd.testing.assert_frame_equal(
        metrics, expected_df, check_dtype=False
    ), "Compute statistics did not return expected values"


@parametrize_with_cases(
    "url, collection, bbox, time, bands, time_aggregate, nodata_fill, exception",
    glob="stac_api_call_fail",
)
def test_stac_api_call_fail(
    url, collection, bbox, time, bands, time_aggregate, nodata_fill, exception
):
    """
    Tests call for stac API fail
    """

    with raises(exception):
        catalog = pystac_client.Client.open(url)

        candidate_items = catalog.search(
            datetime=time[0],
            collections=[collection],
            bbox=bbox,
        ).item_collection()

        _ = get_stac_data(
            stac_items=candidate_items,
            bands=bands,
            time_aggregate=time_aggregate,
            nodata_fill=nodata_fill,
        )
