import numpy as np
from pytest_cases import parametrize_with_cases
import xarray as xr
from pytest import raises
import pandas as pd

from tests.conftest import _compare_metrics_df_with_xarray
from gval.utils.loading_datasets import (
    adjust_memory_strategy,
    get_current_memory_strategy,
    _handle_xarray_memory,
)


@parametrize_with_cases(
    "candidate_map, benchmark_map, positive_categories, negative_categories, rasterize_attributes, memory_strategies, comparison_function",
    glob="data_array_accessor_success",
)
def test_data_array_accessor_success(
    candidate_map,
    benchmark_map,
    positive_categories,
    negative_categories,
    rasterize_attributes,
    memory_strategies,
    comparison_function,
):
    adjust_memory_strategy(memory_strategies)

    assert get_current_memory_strategy() == memory_strategies

    candidate_map = _handle_xarray_memory(candidate_map)

    data = candidate_map.gval.categorical_compare(
        benchmark_map=benchmark_map,
        positive_categories=positive_categories,
        negative_categories=negative_categories,
        rasterize_attributes=rasterize_attributes,
        comparison_function=comparison_function,
        allow_candidate_values=[1, 2, np.nan],
        allow_benchmark_values=[0, 2, np.nan],
    )

    if rasterize_attributes is not None:
        assert isinstance(data[0], pd.DataFrame)
    else:
        assert isinstance(data[0], xr.DataArray)
    assert isinstance(data[1], pd.DataFrame)
    assert isinstance(data[2], pd.DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map, positive_categories, negative_categories, memory_strategies, exception",
    glob="data_array_accessor_fail",
)
def test_data_array_accessor_fail(
    candidate_map,
    benchmark_map,
    positive_categories,
    negative_categories,
    memory_strategies,
    exception,
):
    with raises(exception):
        adjust_memory_strategy(memory_strategies)

        if exception == OSError:
            candidate_map.encoding["source"] = "arb"
            _handle_xarray_memory(candidate_map)

        _ = candidate_map.gval.categorical_compare(
            benchmark_map=benchmark_map,
            positive_categories=positive_categories,
            negative_categories=negative_categories,
        )


@parametrize_with_cases(
    "candidate_map, benchmark_map, rasterize_attributes",
    glob="data_array_accessor_homogenize",
)
def test_data_array_accessor_homogenize(
    candidate_map, benchmark_map, rasterize_attributes
):
    data = candidate_map.gval.homogenize(
        benchmark_map=benchmark_map, rasterize_attributes=rasterize_attributes
    )

    assert isinstance(data[0], xr.DataArray)
    assert isinstance(data[1], xr.DataArray)


@parametrize_with_cases(
    "candidate_map, benchmark_map, vectorized",
    glob="data_array_accessor_compute_agreement",
)
def test_data_array_accessor_compute_agreement_success(
    candidate_map, benchmark_map, vectorized
):
    aligned_cand, aligned_bench = candidate_map.gval.homogenize(
        benchmark_map=benchmark_map
    )

    if vectorized:
        aligned_cand.gval.agreement_map_format = "vector"

    data = aligned_cand.gval.compute_agreement_map(benchmark_map=aligned_bench)

    assert isinstance(data, xr.DataArray)


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_array_accessor_compute_agreement"
)
def test_data_array_accessor_compute_agreement_fail(candidate_map, benchmark_map):
    with raises(ValueError):
        _ = candidate_map.gval.compute_agreement_map(benchmark_map=benchmark_map)


@parametrize_with_cases(
    "agreement_map",
    glob="data_array_accessor_crosstab_table_success",
)
def test_data_array_accessor_crosstab_table_success(agreement_map):
    data = agreement_map.gval.compute_crosstab()

    assert isinstance(data, pd.DataFrame)


@parametrize_with_cases(
    "agreement_map, exception",
    glob="data_array_accessor_crosstab_table_fail",
)
def test_data_array_accessor_crosstab_table_fail(agreement_map, exception):
    with raises(exception):
        _ = agreement_map.gval.compute_crosstab()


@parametrize_with_cases(
    "candidate_map, benchmark_map, positive_categories, negative_categories, rasterize_attributes",
    glob="data_set_accessor_success",
)
def test_data_set_accessor_success(
    candidate_map,
    benchmark_map,
    positive_categories,
    negative_categories,
    rasterize_attributes,
):
    data = candidate_map.gval.categorical_compare(
        benchmark_map=benchmark_map,
        positive_categories=positive_categories,
        negative_categories=negative_categories,
        rasterize_attributes=rasterize_attributes,
    )

    if rasterize_attributes is None:
        assert isinstance(data[0], xr.Dataset)
    else:
        assert isinstance(data[0], pd.DataFrame)
    assert isinstance(data[1], pd.DataFrame)
    assert isinstance(data[2], pd.DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map, rasterize_attributes",
    glob="data_set_accessor_homogenize",
)
def test_data_set_accessor_homogenize(
    candidate_map, benchmark_map, rasterize_attributes
):
    data = candidate_map.gval.homogenize(
        benchmark_map=benchmark_map, rasterize_attributes=rasterize_attributes
    )

    assert isinstance(data[0], xr.Dataset)
    assert isinstance(data[1], xr.Dataset)


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_set_accessor_compute_agreement"
)
def test_data_set_accessor_compute_agreement_success(candidate_map, benchmark_map):
    aligned_cand, aligned_bench = candidate_map.gval.homogenize(
        benchmark_map=benchmark_map
    )
    data = aligned_cand.gval.compute_agreement_map(benchmark_map=aligned_bench)

    assert isinstance(data, xr.Dataset)


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_set_accessor_compute_agreement"
)
def test_data_set_accessor_compute_agreement_fail(candidate_map, benchmark_map):
    with raises(ValueError):
        _ = candidate_map.gval.compute_agreement_map(benchmark_map=benchmark_map)


@parametrize_with_cases(
    "agreement_map", glob="data_set_accessor_crosstab_table_success"
)
def test_data_set_accessor_crosstab_table_success(agreement_map):
    data = agreement_map.gval.compute_crosstab()

    assert isinstance(data, pd.DataFrame)


@parametrize_with_cases(
    "agreement_map",
    glob="data_set_accessor_crosstab_table_fail",
)
def test_data_set_accessor_crosstab_table_fail(agreement_map):
    with raises(KeyError):
        _ = agreement_map.gval.compute_crosstab()


@parametrize_with_cases(
    "crosstab_df, positive_categories, negative_categories",
    glob="data_frame_accessor_compute_metrics",
)
def test_data_frame_accessor_compute_metrics(
    crosstab_df, positive_categories, negative_categories
):
    data = crosstab_df.gval.compute_categorical_metrics(
        positive_categories=positive_categories, negative_categories=negative_categories
    )

    assert isinstance(data, pd.DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map",
    glob="data_array_accessor_continuous",
)
def test_data_array_accessor_continuous(candidate_map, benchmark_map):
    agreement_map, metrics_df = candidate_map.gval.continuous_compare(
        benchmark_map=benchmark_map, metrics="all"
    )

    assert isinstance(agreement_map, xr.DataArray)
    assert isinstance(metrics_df, pd.DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map",
    glob="data_set_accessor_continuous",
)
def test_data_set_accessor_continuous(
    candidate_map,
    benchmark_map,
):
    agreement_map, metrics_df = candidate_map.gval.continuous_compare(
        benchmark_map=benchmark_map, metrics="all"
    )

    assert isinstance(agreement_map, xr.Dataset)
    assert isinstance(metrics_df, pd.DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map, agreement_map",
    glob="accessor_attributes",
)
def test_accessor_attributes(candidate_map, benchmark_map, agreement_map):
    attrs_df, agreement_map = candidate_map.gval.attribute_tracking(
        benchmark_map=benchmark_map, agreement_map=agreement_map
    )

    assert isinstance(agreement_map, xr.DataArray)
    assert isinstance(attrs_df, pd.DataFrame)


@parametrize_with_cases(
    "vector_map, reference_map, attributes",
    glob="dataframe_accessor_rasterize",
)
def test_dataframe_accessor_rasterize(vector_map, reference_map, attributes):
    raster_map = vector_map.gval.rasterize_data(
        reference_map=reference_map, rasterize_attributes=attributes
    )

    assert isinstance(raster_map, type(reference_map))
    assert raster_map.shape == reference_map.shape


@parametrize_with_cases(
    "candidate_map, benchmark_map, compute_kwargs, expected_df",
    glob="data_array_accessor_probabilistic_success",
)
def test_data_array_accessor_probabilistic_success(
    candidate_map, benchmark_map, compute_kwargs, expected_df
):
    _, metrics_df = candidate_map.gval.probabilistic_compare(
        benchmark_map, **compute_kwargs
    )

    _compare_metrics_df_with_xarray(metrics_df, expected_df)
