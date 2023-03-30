from pytest_cases import parametrize_with_cases
import xarray as xr
from pytest import raises
from pandas import DataFrame

from gval.utils.exceptions import RasterMisalignment


@parametrize_with_cases(
    "candidate_map, benchmark_map, positive_categories, negative_categories, dimensions",
    glob="data_array_accessor_success",
)
def test_data_array_accessor_success(
    candidate_map, benchmark_map, positive_categories, negative_categories, dimensions
):
    data = candidate_map.gval.categorical_compare(
        benchmark_map=benchmark_map,
        positive_categories=positive_categories,
        negative_categories=negative_categories,
        dimensions=dimensions,
    )

    assert isinstance(data[0], xr.DataArray)
    assert isinstance(data[1], DataFrame)
    assert isinstance(data[2], DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map, positive_categories, negative_categories, dimensions",
    glob="data_array_accessor_fail",
)
def test_data_array_accessor_fail(
    candidate_map, benchmark_map, positive_categories, negative_categories, dimensions
):
    with raises(ValueError):
        _ = candidate_map.gval.categorical_compare(
            benchmark_map=benchmark_map,
            positive_categories=positive_categories,
            negative_categories=negative_categories,
            dimensions=dimensions,
        )


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_array_accessor_spatial_alignment"
)
def test_data_array_accessor_spatial_alignment(candidate_map, benchmark_map):
    data = candidate_map.gval.spatial_alignment(benchmark_map=benchmark_map)

    assert isinstance(data[0], xr.DataArray)
    assert isinstance(data[1], xr.DataArray)
    assert data[0].gval.aligned
    assert data[1].gval.aligned


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_array_accessor_compute_agreement"
)
def test_data_array_accessor_compute_agreement_success(candidate_map, benchmark_map):
    aligned_cand, aligned_bench = candidate_map.gval.spatial_alignment(
        benchmark_map=benchmark_map
    )
    data = aligned_cand.gval.compute_agreement_map(benchmark_map=aligned_bench)

    assert isinstance(data, xr.DataArray)


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_array_accessor_compute_agreement"
)
def test_data_array_accessor_compute_agreement_fail(candidate_map, benchmark_map):
    with raises(RasterMisalignment):
        _ = candidate_map.gval.compute_agreement_map(benchmark_map=benchmark_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, dimensions",
    glob="data_array_accessor_crosstab_table_success",
)
def test_data_array_accessor_crosstab_table_success(
    candidate_map, benchmark_map, dimensions
):
    aligned_cand, aligned_bench = candidate_map.gval.spatial_alignment(
        benchmark_map=benchmark_map
    )
    data = aligned_cand.gval.compute_crosstab(
        benchmark_map=aligned_bench, dimensions=dimensions
    )

    assert isinstance(data, DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map, dimensions, exception",
    glob="data_array_accessor_crosstab_table_fail",
)
def test_data_array_accessor_crosstab_table_fail(
    candidate_map, benchmark_map, dimensions, exception
):
    if exception != RasterMisalignment:
        candidate_map, benchmark_map = candidate_map.gval.spatial_alignment(
            benchmark_map=benchmark_map
        )

    with raises(exception):
        _ = candidate_map.gval.compute_crosstab(
            benchmark_map=benchmark_map, dimensions=dimensions
        )


@parametrize_with_cases(
    "candidate_map, benchmark_map, positive_categories, negative_categories",
    glob="data_set_accessor_success",
)
def test_data_set_accessor_success(
    candidate_map, benchmark_map, positive_categories, negative_categories
):
    data = candidate_map.gval.categorical_compare(
        benchmark_map=benchmark_map,
        positive_categories=positive_categories,
        negative_categories=negative_categories,
    )

    assert isinstance(data[0], xr.DataSet)
    assert isinstance(data[1], DataFrame)
    assert isinstance(data[2], DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_set_accessor_spatial_alignment"
)
def test_data_set_accessor_spatial_alignment(candidate_map, benchmark_map):
    data = candidate_map.gval.spatial_alignment(benchmark_map=benchmark_map)

    assert isinstance(data[0], xr.DataSet)
    assert isinstance(data[1], xr.DataSet)
    assert data[0].gval.aligned
    assert data[1].gval.aligned


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_set_accessor_compute_agreement"
)
def test_data_set_accessor_compute_agreement_success(candidate_map, benchmark_map):
    aligned_cand, aligned_bench = candidate_map.gval.spatial_alignment(
        benchmark_map=benchmark_map
    )
    data = aligned_cand.gval.compute_agreement_map(benchmark_map=aligned_bench)

    assert isinstance(data, xr.Dataset)


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_set_accessor_compute_agreement"
)
def test_data_set_accessor_compute_agreement_fail(candidate_map, benchmark_map):
    with raises(RasterMisalignment):
        _ = candidate_map.gval.compute_agreement_map(benchmark_map=benchmark_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map", glob="data_set_accessor_crosstab_table_success"
)
def test_data_set_accessor_crosstab_table_success(candidate_map, benchmark_map):
    aligned_cand, aligned_bench = candidate_map.gval.spatial_alignment(
        benchmark_map=benchmark_map
    )
    data = aligned_cand.gval.compute_crosstab(benchmark_map=aligned_bench)

    assert isinstance(data, DataFrame)


@parametrize_with_cases(
    "candidate_map, benchmark_map, exception",
    glob="data_set_accessor_crosstab_table_fail",
)
def test_data_set_accessor_crosstab_table_fail(candidate_map, benchmark_map, exception):
    with raises(exception):
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
