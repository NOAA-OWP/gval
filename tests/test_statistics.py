import numpy as np
from pytest_cases import parametrize_with_cases
from pytest import raises

from gval.statistics.categorical_statistics import (
    CategoricalStatisticsProcessing as CatStat,
)
from gval.statistics.categorical_stat_funcs import CategoricalStatistics as Stat


@parametrize_with_cases("args, expected", glob="categorical_statistics")
def test_categorical_statistics(args, expected):
    """tests categorical statistics functions"""

    Stat.accuracy(*[args[p] for p in CatStat.get_parameters("accuracy")]),
    Stat.critical_success_index(
        *[args[p] for p in CatStat.get_parameters("critical_success_index")]
    ),
    Stat.f_score(*[args[p] for p in CatStat.get_parameters("f_score")]),
    Stat.false_discovery_rate(
        *[args[p] for p in CatStat.get_parameters("false_discovery_rate")]
    ),
    Stat.false_negative_rate(
        *[args[p] for p in CatStat.get_parameters("false_negative_rate")]
    ),
    Stat.false_omission_rate(
        *[args[p] for p in CatStat.get_parameters("false_omission_rate")]
    ),
    Stat.false_positive_rate(
        *[args[p] for p in CatStat.get_parameters("false_positive_rate")]
    ),
    Stat.fowlkes_mallows_index(
        *[args[p] for p in CatStat.get_parameters("fowlkes_mallows_index")]
    ),
    Stat.matthews_correlation_coefficient(
        *[args[p] for p in CatStat.get_parameters("matthews_correlation_coefficient")]
    ),
    Stat.negative_likelihood_ratio(
        *[args[p] for p in CatStat.get_parameters("negative_likelihood_ratio")]
    ),
    Stat.negative_predictive_value(
        *[args[p] for p in CatStat.get_parameters("negative_predictive_value")]
    ),
    Stat.positive_likelihood_ratio(
        *[args[p] for p in CatStat.get_parameters("positive_likelihood_ratio")]
    ),
    Stat.positive_predictive_value(
        *[args[p] for p in CatStat.get_parameters("positive_predictive_value")]
    ),
    Stat.prevalence(*[args[p] for p in CatStat.get_parameters("prevalence")]),
    Stat.prevalence_threshold(
        *[args[p] for p in CatStat.get_parameters("prevalence_threshold")]
    ),
    Stat.true_negative_rate(
        *[args[p] for p in CatStat.get_parameters("true_negative_rate")]
    ),
    Stat.true_positive_rate(
        *[args[p] for p in CatStat.get_parameters("true_positive_rate")]
    )


@parametrize_with_cases("names, args, expected", glob="compute_statistics")
def test_compute_statistics(names, args, expected):
    """tests compute statistics function"""

    results = CatStat.process_statistics(names, args)

    np.testing.assert_equal(
        results, expected
    ), "Compute statistics did not return expected values"


@parametrize_with_cases("names, args, exception", glob="compute_statistics_fail")
def test_compute_statistics_fail(names, args, exception):
    """tests compute statistics fail function"""

    with np.errstate(divide="ignore"):
        with raises(exception):
            stat_names = CatStat.available_functions() if names == "all" else names
            _ = [CatStat.process_statistics(name, args) for name in stat_names]


@parametrize_with_cases("args, func", glob="register_function")
def test_register_function(args, func):
    """tests register func function"""

    CatStat.register_function(**args)(func)


@parametrize_with_cases("args, func, exception", glob="register_function_fail")
def test_register_function_fail(args, func, exception):
    """tests register func fail function"""

    with raises(exception):
        CatStat.register_function(**args)(func)


@parametrize_with_cases("names, args, cls", glob="register_class")
def test_register_class(names, args, cls):
    """tests register class function"""

    CatStat.register_function_class(**args)(cls)

    if [name in CatStat.registered_functions for name in names] != [True] * len(names):
        assert False, "Unable to register all class functions"


@parametrize_with_cases("args, cls, exception", glob="register_class_fail")
def test_register_class_fail(args, cls, exception):
    """tests register class fail function"""

    with raises(exception):
        CatStat.register_function_class(**args)(cls)


@parametrize_with_cases("name, params", glob="get_param")
def test_get_param(name, params):
    """tests get param function"""

    _params = CatStat.get_parameters(name)
    assert _params == params


@parametrize_with_cases("name", glob="get_param_fail")
def test_get_param_fail(name):
    """tests get param fail function"""

    with raises(KeyError):
        _ = CatStat.get_parameters(name)


def test_get_all_param():
    """tests get all params function"""

    try:
        CatStat.get_all_parameters()
    except KeyError:
        assert False, "Signature dict not present or keys changed"
