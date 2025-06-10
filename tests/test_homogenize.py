"""
Test functionality for gval/homogenize/spatial_alignment.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest import raises
from pytest_cases import parametrize_with_cases
import xarray as xr
import numpy as np
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from gval.homogenize.spatial_alignment import (
    _matching_crs,
    _matching_spatial_indices,
    _rasters_intersect,
    _align_rasters,
    _spatial_alignment,
)
from gval.homogenize.numeric_alignment import _align_numeric_data_type
from gval.homogenize.rasterize import _rasterize_data
from gval.homogenize.vectorize import _vectorize_data
from gval.utils.exceptions import RasterMisalignment, RastersDontIntersect


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_match", glob="matching_crs"
)
def test_matching_crs(candidate_map, benchmark_map, expected_match):
    """Tests if two maps have matching CRSs"""
    matching = _matching_crs(candidate_map, benchmark_map)

    assert matching == expected_match, "CRSs of maps are not matching"


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_spatial_indices_matches",
    glob="matching_spatial_indices_success",
)
def test_matching_spatial_indices_success(
    candidate_map, benchmark_map, expected_spatial_indices_matches
):
    """Tests for matching indices in two xarrays"""
    matching = _matching_spatial_indices(candidate_map, benchmark_map)
    assert matching == expected_spatial_indices_matches, "Indices don't match expected"


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_spatial_indices_matches",
    glob="matching_spatial_indices_fail",
)
def test_matching_spatial_indices_fail(
    candidate_map, benchmark_map, expected_spatial_indices_matches
):
    """Tests for matching indices in two xarrays"""
    with raises(RasterMisalignment):
        _matching_spatial_indices(candidate_map, benchmark_map, raise_exception=True)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_intersect",
    glob="rasters_intersect_no_exception",
)
def test_rasters_intersect_no_exception(
    candidate_map, benchmark_map, expected_intersect
):
    """Tests the intersection of rasters"""
    intersect = _rasters_intersect(candidate_map, benchmark_map)
    assert intersect == expected_intersect


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected_intersect",
    glob="rasters_intersect_exception",
)
def test_rasters_intersect_exception(candidate_map, benchmark_map, expected_intersect):
    """Tests the intersection of rasters"""
    with raises(RastersDontIntersect):
        _rasters_intersect(candidate_map, benchmark_map, raise_exception=True)


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs", glob="align_rasters"
)
def test_align_rasters(candidate_map, benchmark_map, target_map, kwargs):
    """Tests the alignment of rasters"""

    # This might raise value errors associated with
    cam, bem = _align_rasters(candidate_map, benchmark_map, target_map, **kwargs)

    # this tests for matching spatial indices
    _matching_spatial_indices(cam, bem, raise_exception=True)


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs", glob="align_rasters_fail"
)
def test_align_rasters_fail(candidate_map, benchmark_map, target_map, kwargs):
    """Tests the alignment of rasters"""

    with raises(ValueError):
        _, _ = _align_rasters(candidate_map, benchmark_map, target_map, **kwargs)


@parametrize_with_cases(
    "candidate_map, benchmark_map, resampling, target_map",
    glob="align_rasters_fail_nodata",
)
def test_align_rasters_fail_nodata(
    candidate_map, benchmark_map, resampling, target_map
):
    """Tests the alignment of rasters"""

    with raises(ValueError):
        _, _ = _align_rasters(candidate_map, benchmark_map, target_map, resampling)


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs",
    glob="spatial_alignment",
)
def test_spatial_alignment(candidate_map, benchmark_map, target_map, kwargs):
    """Tests spatial_alignment function"""

    cam, bem = _spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs)

    try:
        # xr.align raises a value error if coordinates don't align
        xr.align(cam, bem, join="exact")
    except ValueError:
        assert False, "Candidate and benchmark failed to align"


@parametrize_with_cases(
    "candidate_map, benchmark_map, target_map, kwargs",
    glob="spatial_alignment_fail",
)
def test_spatial_alignment_fail(candidate_map, benchmark_map, target_map, kwargs):
    """Tests spatial_alignment function"""

    with raises(RastersDontIntersect):
        _, _ = _spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs)


@parametrize_with_cases(
    "candidate_map, benchmark_map, rasterize_attributes, expected",
    glob="rasterize_vector_success",
)
def test_rasterize_vector_success(
    candidate_map, benchmark_map, rasterize_attributes, expected
):
    """Test rasterize vector success"""

    benchmark_raster = _rasterize_data(
        candidate_map=candidate_map,
        benchmark_map=benchmark_map,
        rasterize_attributes=rasterize_attributes,
    )

    if isinstance(benchmark_raster, xr.Dataset):
        assert benchmark_raster.band_1.shape == candidate_map.band_1.shape
    else:
        assert benchmark_raster.shape == candidate_map.shape

    xr.testing.assert_equal(benchmark_raster, expected)


@parametrize_with_cases(
    "candidate_map, benchmark_map, rasterize_attributes",
    glob="rasterize_vector_fail",
)
def test_rasterize_vector_fail(candidate_map, benchmark_map, rasterize_attributes):
    """Tests rasterize vector fail"""

    with raises(KeyError):
        _, _ = _rasterize_data(
            candidate_map=candidate_map,
            benchmark_map=benchmark_map,
            rasterize_attributes=rasterize_attributes,
        )


@parametrize_with_cases(
    "raster_map, expected",
    glob="vectorize_raster_success",
)
def test_vectorize_raster_success(raster_map, expected):
    """Test vectorize raster success"""

    vector_df = _vectorize_data(raster_data=raster_map)

    assert isinstance(vector_df, gpd.GeoDataFrame)
    assert_geodataframe_equal(
        vector_df.sort_values("geometry", ignore_index=True),
        expected.sort_values("geometry", ignore_index=True),
        check_index_type=False,
        check_dtype=False,
    )


@parametrize_with_cases(
    "raster_map",
    glob="vectorize_raster_fail",
)
def test_vectorize_raster_fail(raster_map):
    """Tests vectorize raster fail"""

    with raises(AttributeError):
        _ = _vectorize_data(raster_map)


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected", glob="numeric_align_dataarrays"
)
def test_numeric_align_dataarrays(candidate_map, benchmark_map, expected):
    """Tests numeric alignment"""

    c, b = _align_numeric_data_type(candidate_map, benchmark_map)

    assert c.data.dtype == expected
    assert b.data.dtype == expected

    assert np.isclose(
        float(
            np.mean(
                xr.where(candidate_map == candidate_map.rio.nodata, 0, candidate_map)
            )
        ),
        float(np.mean(xr.where(c == c.rio.nodata, 0, c))),
    )
    assert np.isclose(
        float(
            np.mean(
                xr.where(benchmark_map == benchmark_map.rio.nodata, 0, benchmark_map)
            )
        ),
        float(np.mean(xr.where(b == b.rio.nodata, 0, b))),
    )


@parametrize_with_cases(
    "candidate_map, benchmark_map, expected", glob="numeric_align_datasets"
)
def test_numeric_align_datasets(candidate_map, benchmark_map, expected):
    """Tests numeric alignment for datasets"""

    c, b = _align_numeric_data_type(candidate_map, benchmark_map)

    for c_var, cand_var, b_var, bench_var in zip(
        c.data_vars, candidate_map.data_vars, b.data_vars, benchmark_map.data_vars
    ):
        assert c[c_var].data.dtype == expected
        assert b[b_var].data.dtype == expected

        assert np.isclose(
            float(
                np.mean(
                    xr.where(
                        candidate_map[cand_var] == candidate_map[cand_var].rio.nodata,
                        0,
                        candidate_map[cand_var],
                    )
                )
            ),
            float(np.mean(xr.where(c[c_var] == c[c_var].rio.nodata, 0, c[c_var]))),
        )

        assert np.isclose(
            float(
                np.mean(
                    xr.where(
                        benchmark_map[bench_var] == benchmark_map[bench_var].rio.nodata,
                        0,
                        benchmark_map[bench_var],
                    )
                )
            ),
            float(np.mean(xr.where(b[b_var] == b[b_var].rio.nodata, 0, b[b_var]))),
        )
