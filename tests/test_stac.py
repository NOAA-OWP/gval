from pytest_cases import parametrize_with_cases
from pytest import raises
import xarray as xr
import pandas as pd

from gval.utils.loading_datasets import get_stac_data

@parametrize_with_cases("url, collection, bbox, time, bands, time_aggregate, nodata_fill, expected_df", 
                        glob="stac_api_call")
def test_stac_api_call(url, collection, bbox, time, bands, time_aggregate, nodata_fill, expected_df):
    """
    Tests call for stac API, (IDK if this data can be mocked, API calls in unit tests are dubious)
    """

    candidate = get_stac_data(
        url=url,
        collection=collection,
        time=time[0],
        bands=bands,
        bbox=bbox,
        time_aggregate=time_aggregate,
        nodata_fill=nodata_fill
    )

    benchmark = get_stac_data(
        url=url,
        collection=collection,
        time=time[1],
        bands=bands,
        bbox=bbox,
        time_aggregate=time_aggregate,
        nodata_fill=nodata_fill
    )

    agreement, metrics = candidate.gval.continuous_compare(benchmark)

    bnds = [bands] if isinstance(bands, str) else bands

    assert isinstance(agreement, xr.Dataset)
    assert bnds == [agreement[var].attrs['original_name'] for var in agreement.data_vars]

    pd.testing.assert_frame_equal(
        metrics, expected_df, check_dtype=False
    ), "Compute statistics did not return expected values"


@parametrize_with_cases("url, collection, bbox, time, bands, time_aggregate, nodata_fill, exception", 
                        glob="stac_api_call_fail")
def test_stac_api_call(url, collection, bbox, time, bands, time_aggregate, nodata_fill, exception):
    """
    Tests call for stac API fail
    """

    with raises(exception):
        _ = get_stac_data(
            url=url,
            collection=collection,
            time=time,
            bands=bands,
            bbox=bbox,
            time_aggregate=time_aggregate,
            nodata_fill=nodata_fill
        )