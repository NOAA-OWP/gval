"""
Continuous Statistics Functions From Error Based Agreement Maps.
"""

from numbers import Number
from typing import Union

import xarray as xr
import numpy as np

from gval.statistics.continuous_stat_utils import convert_output


@convert_output
def mean_absolute_error(error: Union[xr.DataArray, xr.Dataset]) -> Number:
    """
    Compute mean absolute error (MAE).

    Either `error` or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.

    Returns
    -------
    MAE : Number
        Mean absolute error.

    References
    ----------
    .. [1] `Mean absolute error <https://en.wikipedia.org/wiki/Mean_absolute_error>`_
    """
    return np.abs(error).mean()


@convert_output
def mean_squared_error(error: Union[xr.DataArray, xr.Dataset]) -> Number:
    """
    Compute mean squared error (MSE).

    Either `error` or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.

    Returns
    -------
    MSE : Number
        Mean squared error.

    References
    ----------
    .. [1] `Mean squared error <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    """

    return (error**2).mean()


@convert_output
def root_mean_squared_error(error: Union[xr.DataArray, xr.Dataset]) -> Number:
    """
    Compute root mean squared error (RMSE).

    Either `error` or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.

    Returns
    -------
    RMSE : Number
        Root mean squared error.

    References
    ----------
    .. [1] `Root mean square deviation <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_
    """

    return np.sqrt((error**2).mean())


@convert_output
def mean_signed_error(error: Union[xr.DataArray, xr.Dataset]) -> Number:
    """
    Compute mean signed error (MSiE).

    Either `error` or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.

    Returns
    -------
    MSiE : Number
        Mean signed error.

    References
    ----------
    .. [1] `Mean signed error <https://en.wikipedia.org/wiki/Mean_signed_difference>`_
    """

    return error.mean()


@convert_output
def mean_percentage_error(
    error: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> Number:
    """
    Compute mean percentage error (MPE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    MPE : Number
        Mean percentage error.

    References
    ----------
    .. [1] `Mean percentage error <https://en.wikipedia.org/wiki/Mean_percentage_error>`_
    """
    return (error / benchmark_map.mean()).mean()


@convert_output
def mean_absolute_percentage_error(
    error: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> Number:
    """
    Compute mean absolute percentage error (MAPE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    MAPE : Number
        Mean absolute percentage error.

    References
    ----------
    .. [1] `Mean absolute percentage error <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_
    """

    return np.abs(error / benchmark_map).mean()


@convert_output
def mean_normalized_root_mean_squared_error(
    error: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> Number:
    """
    Compute mean normalized root mean squared error (NRMSE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    mNRMSE : Number
        Mean normalized root mean squared error.

    References
    ----------
    .. [1] `Normalized root-mean-square deviation <https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation>`_
    """

    return np.sqrt((error**2).mean()) / benchmark_map.mean()


@convert_output
def range_normalized_root_mean_squared_error(
    error: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> Number:
    """
    Compute range normalized root mean squared error (RNRMSE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    rNRMSE : Number
        Range normalized root mean squared error.

    References
    ----------
    .. [1] `Normalized root-mean-square deviation <https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation>`_
    """
    return np.sqrt((error**2).mean()) / (benchmark_map.max() - benchmark_map.min())


@convert_output
def mean_normalized_mean_absolute_error(
    error: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> Number:
    """
    Compute mean normalized mean absolute error (NMAE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    NMAE : Number
        Normalized mean absolute error.

    References
    ----------
    .. [1] `Normalized mean absolute error <https://en.wikipedia.org/wiki/Mean_absolute_error#Normalized_mean_absolute_error>`_
    """
    return np.abs(error).mean() / benchmark_map.mean()


@convert_output
def range_normalized_mean_absolute_error(
    error: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> Number:
    """
    Compute range normalized mean absolute error (RNMAE).

    Either (`error` and `benchmark_map`) or (`candidate_map` and `benchmark_map`) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    rNMAE : Number
        Range normalized mean absolute error.

    References
    ----------
    .. [1] `Normalized mean absolute error <https://en.wikipedia.org/wiki/Mean_absolute_error#Normalized_mean_absolute_error>`_
    """
    return np.abs(error).mean() / (benchmark_map.max() - benchmark_map.min())


@convert_output
def coefficient_of_determination(
    error: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> Number:
    """
    Compute coefficient of determination (R2).

    Either (error and benchmark_map) or (candidate_map and benchmark_map) must be provided.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    R2 : Number
        Coefficient of determination.

    References
    ----------
    .. [1] `Coefficient of determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    """

    return 1 - (error**2).sum() / ((benchmark_map - benchmark_map.mean()) ** 2).sum()


@convert_output
def symmetric_mean_absolute_percentage_error(
    error: Union[xr.DataArray, xr.Dataset],
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
) -> Number:
    """
    Compute symmetric mean absolute percentage error (sMAPE).

    Both `candidate_map` and `benchmark_map` must be provided. `error` can be provided to avoid recomputing it.

    Parameters
    ----------
    error : Union[xr.DataArray, xr.Dataset]
        Candidate minus benchmark error.
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map.

    Returns
    -------
    sMAPE : Number
        Symmetric mean absolute percentage error.

    References
    ----------
    .. [1] `Symmetric mean absolute percentage error <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_
    """
    return (
        2 * np.abs(error).sum() / (np.abs(candidate_map) + np.abs(benchmark_map)).sum()
    )
