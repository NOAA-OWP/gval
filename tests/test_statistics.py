import numpy as np
from pytest_cases import parametrize_with_cases
from pytest import raises

from gval.statistics.categorical_statistics import CategoricalStatistics

cat_stat = CategoricalStatistics()


@parametrize_with_cases("names, args, funcs, expected", glob="categorical_statistics")
def test_categorical_statistics(names, args, funcs, expected):
    """tests categorical statistics functions"""

    np.testing.assert_equal(
        [
            func(*[args[p] for p in cat_stat.get_parameters(name)])
            for func, name in zip(funcs, names)
        ],
        expected,
    ), "Compute statistics did not return expected values"


@parametrize_with_cases("names, args, expected", glob="compute_statistics")
def test_compute_statistics(names, args, expected):
    """tests compute statistics fail function"""

    # NOTE: Removed bc this should be handled within process_statistics.
    # stat_names = cat_stat.available_functions() if names == "all" else names

    np.testing.assert_equal(
        cat_stat.process_statistics(names, **args)[
            0
        ],  # NOTE: Only testing metric value returns.
        expected,
    ), "Compute statistics did not return expected values"


@parametrize_with_cases("names, args, exception", glob="compute_statistics_fail")
def test_compute_statistics_fail(names, args, exception):
    """tests compute statistics fail function"""

    with np.errstate(divide="ignore"):
        with raises(exception):
            # NOTE: Removed bc this should be handled within process_statistics.
            # stat_names = cat_stat.available_functions() if names == "all" else names
            cat_stat.process_statistics(names, **args)


@parametrize_with_cases("args, func", glob="register_function")
def test_register_function(args, func):
    """tests register func function"""

    cat_stat.register_function(**args)(func)


@parametrize_with_cases("args, func, exception", glob="register_function_fail")
def test_register_function_fail(args, func, exception):
    """tests register func fail function"""

    with raises(exception):
        cat_stat.register_function(**args)(func)


@parametrize_with_cases("names, args, cls", glob="register_class")
def test_register_class(names, args, cls):
    """tests register class function"""

    cat_stat.register_function_class(**args)(cls)

    if [name in cat_stat.registered_functions for name in names] != [True] * len(names):
        assert False, "Unable to register all class functions"


@parametrize_with_cases("args, cls, exception", glob="register_class_fail")
def test_register_class_fail(args, cls, exception):
    """tests register class fail function"""

    with raises(exception):
        cat_stat.register_function_class(**args)(cls)


@parametrize_with_cases("name, params", glob="get_param")
def test_get_param(name, params):
    """tests get param function"""

    _params = cat_stat.get_parameters(name)
    assert _params == params


@parametrize_with_cases("name", glob="get_param_fail")
def test_get_param_fail(name):
    """tests get param fail function"""

    with raises(KeyError):
        _ = cat_stat.get_parameters(name)


def test_get_all_param():
    """tests get all params function"""

    try:
        cat_stat.get_all_parameters()
    except KeyError:
        assert False, "Signature dict not present or keys changed"


def test_available_functions():
    """tests get available functions"""

    a_funcs = cat_stat.available_functions()

    assert isinstance(a_funcs, list)
