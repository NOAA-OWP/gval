"""
Test functionality 
"""

#__all__ = ['*']
__author__ = 'Fernando Aristizabal'

import pytest
import os
import sys

# temporary
sys.path.append(os.path.abspath('..'))

from gval.utils.loading_datasets import load_raster_as_xarray
from gval.prep_comparison.spatial_alignment import *
from gval.compare import *
from gval.compare import _are_not_natural_numbers
from config import TEST_DATA

test_data_dir = TEST_DATA

@pytest.fixture(scope='module', params=range(1))
def candidate_map_fp(request):
    """ returns candidate maps """
    filepath = os.path.join(test_data_dir,f'candidate_map_{request.param}.tif')
    yield filepath

@pytest.fixture(scope='module', params=range(1))
def benchmark_map_fp(request):
    """ returns benchmark maps """
    filepath = os.path.join(test_data_dir,f'benchmark_map_{request.param}.tif')
    yield filepath

@pytest.fixture(scope='module')
def candidate_map(candidate_map_fp):
    """ returns candidate maps """
    yield load_raster_as_xarray(candidate_map_fp)

@pytest.fixture(scope='module')
def benchmark_map(benchmark_map_fp):
    """ returns benchmark maps """
    yield load_raster_as_xarray(benchmark_map_fp)

def test_load_candidate_as_xarray(candidate_map_fp):
    """ tests loading candidate raster as xarray DataArray """
    candidate_map = load_raster_as_xarray(candidate_map_fp)
    assert isinstance(candidate_map,xarray.DataArray), "candidate_map is not an xarray.DataArray"

def test_load_benchmark_as_xarray(benchmark_map_fp):
    """ tests loading benchmark raster as xarray DataArray """
    benchmark_map = load_raster_as_xarray(benchmark_map_fp)
    assert isinstance(benchmark_map,xarray.DataArray), "benchmark_map is not an xarray.DataArray"

@pytest.fixture(scope='module', params=[True])
def expect_matching_crs(request):
    """ Returns expect value for matching CRS test """
    yield request.param

def test_matching_crs(candidate_map, benchmark_map, expect_matching_crs):
    """ Tests for matching CRSs """
    matching = matching_crs(candidate_map,benchmark_map)
    assert matching == expect_matching_crs, f'matching_crs result ({matching}) does not agree with expected ({expect_matching_crs})'

@pytest.fixture(scope='function', params=[False])
def expect_matching_spatial_indices(request):
    """ Returns expect value for matching indices test """
    yield request.param

def test_matching_spatial_indices(candidate_map, benchmark_map, expect_matching_spatial_indices):
    """ Tests for matching indices in two xarray DataArrays """
    matching = matching_spatial_indices(candidate_map, benchmark_map)
    assert matching == expect_matching_spatial_indices, f'Expected {expect_matching_indices} while matching indices {matching}'

@pytest.fixture(scope='module', params=['benchmark','candidate'])
def target_map(request):
    """ Target map fixture """
    yield request.param

@pytest.fixture(scope='module', params=[None,'EPSG:4329'])
def dst_crs(request):
    """ dst_crs fixture """
    yield request.param

def test_transform_bounds(candidate_map, benchmark_map, target_map, dst_crs):
    """ Tests the transformation of bounds given a target map or dst_crs """
    tb = transform_bounds(candidate_map,benchmark_map,target_map, dst_crs)
    assert isinstance(tb,tuple), f"{tb} is not tuple type"

def test_rasters_intersect(candidate_map, benchmark_map):
    """ Tests the intersection of rasters """
    intersect = rasters_intersect(candidate_map.rio.bounds(), benchmark_map.rio.bounds())
    assert intersect, "Maps don't spatially intersect"

@pytest.fixture(scope='function', params=[True])
def expect_checks_for_single_band(request):
    """ checks for single band expects fixture """
    yield request.param

def test_checks_for_single_band(candidate_map,benchmark_map,expect_checks_for_single_band):
    """ Tests checks for single band fixture """
    bsb = checks_for_single_band(candidate_map,benchmark_map)
    assert bsb == expect_checks_for_single_band, "Both candidate and benchmark expected to be single band"

@pytest.fixture(scope='module', params=[None,{'dst_crs':'EPSG:4329'}])
def kwargs(request):
    """ kwargs fixture """
    yield request.param

def test_align_rasters(candidate_map, benchmark_map, target_map, **kwargs):
    """ Tests the alignment of rasters """
    cam, bem = align_rasters(candidate_map, benchmark_map, target_map, **kwargs)
    assert isinstance(cam,xarray.DataArray), "Aligned candidate raster not xarray DataArray"
    assert isinstance(bem,xarray.DataArray), "Aligned benchmark raster not xarray DataArray"

def test_spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs):
    """ Tests spatial_alignment function """
    cam, bem = Spatial_alignment(candidate_map, benchmark_map, target_map, **kwargs)
    assert isinstance(cam,xarray.DataArray), "Aligned candidate raster not xarray DataArray"
    assert isinstance(bem,xarray.DataArray), "Aligned benchmark raster not xarray DataArray"

@pytest.fixture(scope='function', params=[(-1,0,2),(1.09,-13,106),(6.090,-10,27),(-10.39023,13,196)])
def pairs_for_natural_numbers(request):
    """ makes candidate """
    yield request.param

def test_are_not_natural_numbers(pairs_for_natural_numbers):
    c, b, a = pairs_for_natural_numbers
    with pytest.raises(ValueError) as exc:
        _are_not_natural_numbers(c,b)

@pytest.fixture(scope='function', params=[(1,0,1),(1,13,118),(6,0,21),(6,13,203)])
def cantor_pair_input(request):
    """ makes candidate """
    yield request.param

def test_cantor_pair(cantor_pair_input):
    """ tests cantor pairing function """
    c, b, a = cantor_pair_input
    np.testing.assert_equal(cantor_pair(c, b), a), \
        "Cantor function output does not match expected value"

@pytest.fixture(scope='function', params=[(-1,0,1),(1,-13,403),(-6,0,66),(6,-130,37115),(np.nan,-130,np.nan)])
def cantor_pair_signed_input(request):
    """ makes candidate """
    yield request.param

def test_cantor_pair_signed(cantor_pair_signed_input):
    """ tests cantor pairing function """
    c, b, a = cantor_pair_signed_input
    np.testing.assert_equal(cantor_pair_signed(c, b), a), \
        "Signed cantor function output does not match expected value"

@pytest.fixture(scope='function', params=[(1,0,2),(1,13,170),(6,0,42),(6,13,175)])
def szudzik_pair_input(request):
    """ makes candidate """
    yield request.param

def test_szudzik_pair(szudzik_pair_input):
    """ tests szudzikpairing function """
    c, b, a = szudzik_pair_input
    np.testing.assert_equal(szudzik_pair(c, b), a), \
        "szudzik function output does not match expected value"

@pytest.fixture(scope='function', params=[(-1,0,2),(1,-13,627),(-6,0,132),(6,-130,67093),(np.nan,-130,np.nan)])
def szudzik_pair_signed_input(request):
    """ makes candidate """
    yield request.param

def test_szudzik_pair_signed(szudzik_pair_signed_input):
    """ tests szudzik pairing function """
    c, b, a = szudzik_pair_signed_input
    np.testing.assert_equal(szudzik_pair_signed(c, b), a), \
        "Signed szudzik function output does not match expected value"
    
@pytest.fixture(scope='function', params=[2,2,2,2,2])
def candidate_and_benchmark(request):
    """ makes candidate """
    yield np.random.randint(0,100,request.param), np.random.randint(0,100,request.param)

@pytest.fixture(scope='function', params=[cantor_pair_signed])
def comparison_function(request):
    """ makes comparison function """
    yield request.param

def test_compute_agreement_numpy(candidate_and_benchmark, comparison_function):
    """ Tests compute_agreement_numpy """
    candidate, benchmark = candidate_and_benchmark
    agreement = compute_agreement_numpy(candidate, benchmark, comparison_function)
    print(agreement)
