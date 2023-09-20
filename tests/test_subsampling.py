"""
Test functionality for gval/compare.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np
import pandas as pd
from pytest_cases import parametrize_with_cases
from pytest import raises

from gval.subsampling.subsampling import subsample


@parametrize_with_cases("df, args", glob="create_sampling_dataframes")
def test_create_sampling_dataframes(df, args):
    df_out = df.gval.create_subsampling_df(**args)
    args["inplace"] = True
    df.gval.create_subsampling_df(**args)

    assert np.sum(np.sum(df != df_out)) == 0

    if args["subsampling_weights"] is not None:
        assert "weights" in df.columns


@parametrize_with_cases("df, args", glob="create_sampling_dataframes_fail")
def test_create_sampling_dataframes_fail(df, args):
    df.crs = None
    df.drop(columns={"geometry"}, inplace=True)

    with raises(TypeError):
        _ = df.gval.create_subsampling_df(**args)


@parametrize_with_cases(
    "candidate, benchmark, subsample_df, subsample_type, expected_length, sample_percent",
    glob="subsampling",
)
def test_subsampling(
    candidate, benchmark, subsample_df, subsample_type, expected_length, sample_percent
):
    subsample_df.gval.create_subsampling_df(
        subsampling_type=subsample_type, inplace=True
    )

    results = subsample(
        candidate=candidate,
        benchmark=benchmark,
        subsampling_df=subsample_df,
    )

    assert len(results) == expected_length
    for res, perc in zip(results, sample_percent):
        res[0].attrs["sample_percentage"] == perc


@parametrize_with_cases(
    "candidate, benchmark, subsample_df, exception", glob="subsampling_fail"
)
def test_subsampling_fail(candidate, benchmark, subsample_df, exception):
    if exception == ValueError:
        subsample_df.crs = None
    else:
        subsample_df.gval.create_subsampling_df(
            subsampling_type="include", inplace=True
        )

    with raises(exception):
        _ = subsample(
            candidate=candidate, benchmark=benchmark, subsampling_df=subsample_df
        )


@parametrize_with_cases(
    "candidate, benchmark, subsample_df, expected_df, sampling_average",
    glob="categorical_subsample",
)
def test_categorical_subsampling(
    candidate, benchmark, subsample_df, expected_df, sampling_average
):
    subsample_df.gval.create_subsampling_df(subsampling_type="include", inplace=True)

    ag, ctab, met = candidate.gval.categorical_compare(
        benchmark_map=benchmark,
        positive_categories=[2],
        negative_categories=[0, 1],
        subsampling_df=subsample_df,
        subsampling_average=sampling_average,
    )

    pd.testing.assert_frame_equal(
        met, expected_df, check_dtype=False
    ), "Compute statistics did not return expected values"

    assert len(ag) == 2


@parametrize_with_cases(
    "candidate, benchmark, subsample_df, expected_df, sampling_average",
    glob="continuous_subsample",
)
def test_continuous_subsampling(
    candidate, benchmark, subsample_df, expected_df, sampling_average
):
    subsample_df.gval.create_subsampling_df(
        subsampling_type="exclude",
        subsampling_weights=[2, 1],
        inplace=True,
        crs="EPSG:4326",
    )

    ag, met = candidate.gval.continuous_compare(
        benchmark_map=benchmark,
        metrics=["mean_percentage_error"],
        subsampling_df=subsample_df,
        subsampling_average=sampling_average,
    )

    pd.testing.assert_frame_equal(
        met, expected_df, check_dtype=False
    ), "Compute statistics did not return expected values"

    assert len(ag) == 2
