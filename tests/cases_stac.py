"""
Test functionality for gval/statistics modules
"""

# __all__ = ['*']


import numpy as np
import pandas as pd
from pytest_cases import parametrize

url = "https://earth-search.aws.element84.com/v1"
collection = "sentinel-2-l2a"
bbox = [-105.78, 35.79, -105.72, 35.84]


bands = ["aot", ["aot"], ["aot"], ["aot"], ["aot", "rededge1"]]


time = [
    ["2020-04-01", "2020-04-03"],
    ["2020-04-01/2020-04-03", "2020-04-06/2020-04-08"],
    ["2020-04-01/2020-04-03", "2020-04-06/2020-04-08"],
    ["2020-04-01/2020-04-03", "2020-04-06/2020-04-08"],
    ["2020-04-01", "2020-04-03"],
]

time_aggreagte = [None, "mean", "min", "max", None]

nodata_fill = [None, None, None, None, 0]

expected_df = [
    {
        "band": ["1"],
        "coefficient_of_determination": [-1.4498682989135663],
        "mean_absolute_error": [0.025393391237534115],
        "mean_absolute_percentage_error": [0.20440844788169352],
        "mean_normalized_mean_absolute_error": [0.20592052600829783],
        "mean_normalized_root_mean_squared_error": [0.2110757427848459],
        "mean_percentage_error": [-0.20540028455073725],
        "mean_signed_error": [-0.025329236900295846],
        "mean_squared_error": [0.0006775147935436701],
        "range_normalized_mean_absolute_error": [0.31741739046917644],
        "range_normalized_root_mean_squared_error": [0.32536392930255564],
        "root_mean_squared_error": [0.026029114344204452],
        "symmetric_mean_absolute_percentage_error": [0.22778286629345662],
    },
    {
        "band": ["1"],
        "coefficient_of_determination": [0.5954048574471591],
        "mean_absolute_error": [0.011375152039973328],
        "mean_absolute_percentage_error": [0.09299193429857326],
        "mean_normalized_mean_absolute_error": [0.09198102377977661],
        "mean_normalized_root_mean_squared_error": [0.0950603218421001],
        "mean_percentage_error": [-0.03739148233378963],
        "mean_signed_error": [-0.004624147232424582],
        "mean_squared_error": [0.0001382026920448174],
        "range_normalized_mean_absolute_error": [0.10036898858799995],
        "range_normalized_root_mean_squared_error": [0.10372909504665788],
        "root_mean_squared_error": [0.011755964105287894],
        "symmetric_mean_absolute_percentage_error": [0.09373343991103603],
    },
    {
        "band": ["1"],
        "coefficient_of_determination": [0.62257119465569],
        "mean_absolute_error": [0.017518999671533943],
        "mean_absolute_percentage_error": [0.19568589409159515],
        "mean_normalized_mean_absolute_error": [0.15469671510079602],
        "mean_normalized_root_mean_squared_error": [0.1715846316993916],
        "mean_percentage_error": [0.013368590460465646],
        "mean_signed_error": [0.0015139580160649765],
        "mean_squared_error": [0.0003775836662784796],
        "range_normalized_mean_absolute_error": [0.11450326582701924],
        "range_normalized_root_mean_squared_error": [0.12700334769555513],
        "root_mean_squared_error": [0.019431512197419933],
        "symmetric_mean_absolute_percentage_error": [0.15366954251075923],
    },
    {
        "band": ["1"],
        "coefficient_of_determination": [-0.41271964773267755],
        "mean_absolute_error": [0.011908415300546453],
        "mean_absolute_percentage_error": [0.0882205870282268],
        "mean_normalized_mean_absolute_error": [0.08814209475433238],
        "mean_normalized_root_mean_squared_error": [0.08996232968540896],
        "mean_percentage_error": [-0.08717281243113986],
        "mean_signed_error": [-0.011777460658723765],
        "mean_squared_error": [0.00014772792439308438],
        "range_normalized_mean_absolute_error": [0.16772415916262606],
        "range_normalized_root_mean_squared_error": [0.17118785462101266],
        "root_mean_squared_error": [0.0121543376780919],
        "symmetric_mean_absolute_percentage_error": [0.09215897319648417],
    },  # One more for a multi band example
    {
        "band": ["1", "2"],
        "coefficient_of_determination": [-1.4498682989135663, -0.7514062048150743],
        "mean_absolute_error": [0.025393391237534115, 0.10266655092378396],
        "mean_absolute_percentage_error": [0.20440844788169352, np.inf],
        "mean_normalized_mean_absolute_error": [
            0.20592052600829783,
            1.5904259451440834,
        ],
        "mean_normalized_root_mean_squared_error": [
            0.2110757427848459,
            2.8406238608292003,
        ],
        "mean_percentage_error": [-0.20540028455073725, 0.8574267097258231],
        "mean_signed_error": [-0.025329236900295846, 0.05534935042165941],
        "mean_squared_error": [0.0006775147935436701, 0.03362470648587737],
        "range_normalized_mean_absolute_error": [
            0.31741739046917644,
            0.06341747539921179,
        ],
        "range_normalized_root_mean_squared_error": [
            0.32536392930255564,
            0.11326852052594609,
        ],
        "root_mean_squared_error": [0.026029114344204452, 0.18337040787945413],
        "symmetric_mean_absolute_percentage_error": [
            0.22778286629345662,
            0.8343404150271029,
        ],
    },
]


@parametrize(
    "url, collection, bbox, time, bands, time_aggregate, nodata_fill, expected_df",
    list(
        zip(
            [url] * len(time),
            [collection] * len(time),
            [bbox] * len(time),
            time,
            bands,
            time_aggreagte,
            nodata_fill,
            expected_df,
        )
    ),
)
def case_stac_api_call(
    url, collection, bbox, time, bands, time_aggregate, nodata_fill, expected_df
):
    return (
        url,
        collection,
        bbox,
        time,
        bands,
        time_aggregate,
        nodata_fill,
        pd.DataFrame(expected_df),
    )


bands_fail = [["aot"]]

time_fail = [
    ["2020-04-01/2020-04-03", "2020-04-06/2020-04-08"],
]

time_aggregate_fail = [None]

nodata_fill_fail = [None]

exceptions = [ValueError]


@parametrize(
    "url, collection, bbox, time, bands, time_aggregate, nodata_fill, exception",
    list(
        zip(
            [url] * len(time_fail),
            [collection] * len(time_fail),
            [bbox] * len(time_fail),
            time_fail,
            bands_fail,
            time_aggregate_fail,
            nodata_fill_fail,
            exceptions,
        )
    ),
)
def case_stac_api_call_fail(
    url, collection, bbox, time, bands, time_aggregate, nodata_fill, exception
):
    return url, collection, bbox, time, bands, time_aggregate, nodata_fill, exception
