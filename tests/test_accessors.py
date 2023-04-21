from pytest_cases import parametrize_with_cases
import xarray as xr
from pytest import raises
from pandas import DataFrame


@parametrize_with_cases(
    "candidate_map, benchmark_map, positive_categories, negative_categories, rasterize_attributes",
    glob="data_array_accessor_success",
)
def test_data_array_accessor_success(
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

    assert isinstance(data[0], xr.DataArray)
    assert isinstance(data[1], DataFrame)
    assert isinstance(data[2], DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map, positive_categories, negative_categories",
    glob="data_array_accessor_fail",
)
def test_data_array_accessor_fail(
    candidate_map, benchmark_map, positive_categories, negative_categories
):
    with raises(TypeError):
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
    "candidate_map, benchmark_map", glob="data_array_accessor_compute_agreement"
)
def test_data_array_accessor_compute_agreement_success(candidate_map, benchmark_map):
    aligned_cand, aligned_bench = candidate_map.gval.homogenize(
        benchmark_map=benchmark_map
    )
    data = aligned_cand.gval.compute_agreement_map(benchmark_map=aligned_bench)

    assert isinstance(data, xr.DataArray)


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_array_accessor_compute_agreement"
)
def test_data_array_accessor_compute_agreement_fail(candidate_map, benchmark_map):
    with raises(ValueError):
        _ = candidate_map.gval.compute_agreement_map(benchmark_map=benchmark_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map",
    glob="data_array_accessor_crosstab_table_success",
)
def test_data_array_accessor_crosstab_table_success(candidate_map, benchmark_map):
    aligned_cand, aligned_bench = candidate_map.gval.homogenize(
        benchmark_map=benchmark_map
    )
    data = aligned_cand.gval.compute_crosstab(benchmark_map=aligned_bench)

    assert isinstance(data, DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map, exception",
    glob="data_array_accessor_crosstab_table_fail",
)
def test_data_array_accessor_crosstab_table_fail(
    candidate_map, benchmark_map, exception
):
    with raises(exception):
        _ = candidate_map.gval.compute_crosstab(benchmark_map=benchmark_map)


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

    assert isinstance(data[0], xr.Dataset)
    assert isinstance(data[1], DataFrame)
    assert isinstance(data[2], DataFrame)


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
    "candidate_map, benchmark_map", glob="data_set_accessor_crosstab_table_success"
)
def test_data_set_accessor_crosstab_table_success(candidate_map, benchmark_map):
    aligned_cand, aligned_bench = candidate_map.gval.homogenize(
        benchmark_map=benchmark_map
    )
    data = aligned_cand.gval.compute_crosstab(benchmark_map=aligned_bench)

    assert isinstance(data, DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map",
    glob="data_set_accessor_crosstab_table_fail",
)
def test_data_set_accessor_crosstab_table_fail(candidate_map, benchmark_map):
    with raises(IndexError):
        _ = candidate_map.gval.compute_crosstab(benchmark_map=benchmark_map)


@parametrize_with_cases(
    "crosstab_df, positive_categories, negative_categories",
    glob="data_frame_accessor_compute_metrics",
)
def test_data_frame_accessor_compute_metrics(
    crosstab_df, positive_categories, negative_categories
):
    data = crosstab_df.gval.compute_metrics(
        positive_categories=positive_categories, negative_categories=negative_categories
    )

    assert isinstance(data, DataFrame)
