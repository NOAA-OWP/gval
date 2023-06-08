"""
Continuous Statistics Functions From Error Based Agreement Maps.
"""

from numbers import Number
from typing import Union, Optional

import xarray as xr
import numpy as np

from gval.statistics.continuous_stat_utils import convert_output, compute_error_if_none


@compute_error_if_none
@convert_output
def mean_absolute_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute mean absolute error (MAE).

    Either `error` or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    MAE : Number
        Mean absolute error.

    References
    ----------
    .. [1] [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
    """
    return np.abs(error).mean()


@compute_error_if_none
@convert_output
def mean_squared_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute mean squared error (MSE).

    Either `error` or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    MSE : Number
        Mean squared error.

    References
    ----------
    .. [1] [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
    """

    return (error**2).mean()


@compute_error_if_none
@convert_output
def root_mean_squared_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute root mean squared error (RMSE).

    Either `error` or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    RMSE : Number
        Root mean squared error.

    References
    ----------
    .. [1] [Root mean square deviation](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
    """

    return np.sqrt((error**2).mean())


@compute_error_if_none
@convert_output
def mean_signed_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute mean signed error (MSiE).

    Either `error` or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    MSiE : Number
        Mean signed error.

    References
    ----------
    .. [1] [Mean signed error](https://en.wikipedia.org/wiki/Mean_signed_difference)
    """

    return error.mean()


@compute_error_if_none
@convert_output
def mean_percentage_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute mean percentage error (MPE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    MPE : Number
        Mean percentage error.

    References
    ----------
    .. [1] [Mean percentage error](https://en.wikipedia.org/wiki/Mean_percentage_error)
    """
    return (error / benchmark_map.mean()).mean()


@compute_error_if_none
@convert_output
def mean_absolute_percentage_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute mean absolute percentage error (MAPE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    MAPE : Number
        Mean absolute percentage error.

    References
    ----------
    .. [1] [Mean absolute percentage error](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
    """

    return np.abs(error / benchmark_map).mean()


@compute_error_if_none
@convert_output
def mean_normalized_root_mean_squared_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute mean normalized root mean squared error (NRMSE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    mNRMSE : Number
        Mean normalized root mean squared error.

    References
    ----------
    .. [1] [Normalized root-mean-square deviation](https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation)
    """

    return np.sqrt((error**2).mean()) / benchmark_map.mean()


@compute_error_if_none
@convert_output
def range_normalized_root_mean_squared_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute range normalized root mean squared error (RNRMSE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    rNRMSE : Number
        Range normalized root mean squared error.

    References
    ----------
    .. [1] [Normalized root-mean-square deviation](https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation)
    """
    return np.sqrt((error**2).mean()) / (benchmark_map.max() - benchmark_map.min())


@compute_error_if_none
@convert_output
def mean_normalized_mean_absolute_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute mean normalized mean absolute error (NMAE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    NMAE : Number
        Normalized mean absolute error.

    References
    ----------
    .. [1] [Normalized mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error#Normalized_mean_absolute_error)
    """
    return np.abs(error).mean() / benchmark_map.mean()


@compute_error_if_none
@convert_output
def range_normalized_mean_absolute_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute range normalized mean absolute error (RNMAE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    rNMAE : Number
        Range normalized mean absolute error.

    References
    ----------
    .. [1] [Normalized mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error#Normalized_mean_absolute_error)
    """
    return np.abs(error).mean() / (benchmark_map.max() - benchmark_map.min())


@compute_error_if_none
@convert_output
def coefficient_of_determination(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute coefficient of determination (R2).

    Either (error and benchmark_map) or (candidate_map and benchmark_map) must be provided.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    R2 : Number
        Coefficient of determination.

    References
    ----------
    .. [1] [Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)
    """

    return 1 - (error**2).sum() / ((benchmark_map - benchmark_map.mean()) ** 2).sum()


@compute_error_if_none
@convert_output
def symmetric_mean_absolute_percentage_error(
    error: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    benchmark_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> Number:
    """
    Compute symmetric mean absolute percentage error (sMAPE).

    Both `candidate_map` and `benchmark_map` must be provided. `error` can be provided to avoid recomputing it.

    Parameters
    ----------
    error : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate minus benchmark error.
    candidate_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Candidate map.
    benchmark_map : Optional[Union[xr.DataArray, xr.Dataset]], default = None
        Benchmark map.

    Returns
    -------
    sMAPE : Number
        Symmetric mean absolute percentage error.

    References
    ----------
    .. [1] [Symmetric mean absolute percentage error](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
    """
    return (
        2 * np.abs(error).sum() / (np.abs(candidate_map) + np.abs(benchmark_map)).sum()
    )
