"""
Test functionality for gval/statistics modules
"""

# __all__ = ['*']

from pytest_cases import parametrize
from numba.core.errors import TypingError

import gval.statistics.categorical_stat_funcs as cat_stats

func_names = [
    fn for fn in dir(cat_stats) if len(fn) > 5 and "__" not in fn and "Number" not in fn
]
stat_funcs = [getattr(cat_stats, name) for name in func_names]


arg_dicts = [
    {"tp": 120, "tn": 30, "fp": 10, "fn": 10},
    {"tp": 120, "tn": 30, "fp": 10, "fn": 10},
    {"tp": 45, "tn": 20, "fp": 30, "fn": 40},
    {"tp": 45, "fp": 30, "fn": 10},
]

expected_results = [
    [
        0.8823529411764706,
        0.8571428571428571,
        0.9230769230769231,
        0.07692307692307693,
        0.07692307692307693,
        0.25,
        0.25,
        0.9230769230769231,
        0.6730769230769231,
        0.10256410256410257,
        0.75,
        3.6923076923076925,
        0.9230769230769231,
        0.7647058823529411,
        0.3422854855312455,
        0.75,
        0.9230769230769231,
    ],
    [0.8823529411764706],
    [0.48148148148148145, 0.391304347826087, 0.5625],
    [
        0.5294117647058824,
        0.6923076923076923,
        0.4,
        0.18181818181818182,
        0.7006490497453707,
        0.6,
        0.8181818181818182,
    ],
]

stat_names = [
    "all",
    "accuracy",
    ["accuracy", "critical_success_index", "f_score"],
    "all",
]


@parametrize(
    "names, args, funcs, expected",
    list(zip([func_names], arg_dicts[0:1], [stat_funcs], expected_results[0:1])),
)
def case_categorical_statistics(names, args, funcs, expected):
    return names, args, funcs, expected


@parametrize(
    "names, args, expected", list(zip(stat_names, arg_dicts, expected_results))
)
def case_compute_statistics(names, args, expected):
    return names, args, expected


stat_names = ["all", "all", "all", "non_existent_function"]
arg_dicts_fail = [
    {"tp": 120, "tn": 0, "fp": 10, "fn": 10},
    {},
    {"tp": 45, "tn": "test", "fp": "test", "fn": True},
    {"tp": 120, "tn": 0, "fp": 10, "fn": 10},
]
exceptions = [ValueError, ValueError, TypingError, KeyError]


@parametrize(
    "names, args, exception", list(zip(stat_names, arg_dicts_fail, exceptions))
)
def case_compute_statistics_fail(names, args, exception):
    return names, args, exception


stat_args = [{"name": "test_func"}, {"name": "test_func2", "vectorize_func": True}]


def pass1(tp: float, fn: int) -> float:
    return tp + fn


def pass2(tp: float, tn: float, fp: float, fn: float) -> float:
    return tp + tn + fp + fn


stat_funcs = [pass1, pass2]


@parametrize("args, func", list(zip(stat_args, stat_funcs)))
def case_register_function(args, func):
    return args, func


stat_args = [
    {"name": "accuracy"},
    {"name": "test2"},
    {"name": "test2"},
    {"name": "test2"},
    {"name": "test2"},
]


def fail1(tp: float, fn: int) -> float:
    return tp + fn


def fail2(arb: float, arb2: int) -> float:
    return arb + arb2


def fail3(tp: str, fn: int) -> float:
    return tp + fn


def fail4(tp: float, fn: int) -> str:
    return tp + fn


def fail5(tp: float) -> float:
    return tp


stat_fail_funcs = [fail1, fail2, fail3, fail4, fail5]
exceptions = [KeyError, TypeError, TypeError, TypeError, TypeError]


@parametrize("args, func, exception", list(zip(stat_args, stat_fail_funcs, exceptions)))
def case_register_function_fail(args, func, exception):
    return args, func, exception


class Tester:
    @staticmethod
    def pass5(tp: int, fn: int) -> float:
        return tp + fn

    @staticmethod
    def pass6(tp: int, fn: int) -> float:
        return tp + fn


class Tester2:
    @staticmethod
    def pass7(tp: int, fn: int) -> float:
        return tp + fn

    @staticmethod
    def pass8(tp: int, fn: int) -> float:
        return tp + fn


stat_names = [["pass5", "pass6"], ["pass7", "pass8"]]
stat_args = [{}, {"vectorize_func": True}]
stat_class = [Tester, Tester2]


@parametrize("names, args, cls", list(zip(stat_names, stat_args, stat_class)))
def case_register_class(names, args, cls):
    return names, args, cls


class TesterFail1:
    @staticmethod
    def fail6(rp: int, fn: int) -> float:
        return rp + fn


class TesterFail2:
    @staticmethod
    def accuracy(tp: int, fn: int) -> float:
        return tp + fn


stat_args = [{}, {"vectorize_func": True}]
stat_class = [TesterFail1, TesterFail2]
exceptions = [TypeError, KeyError]


@parametrize("args, cls, exception", list(zip(stat_args, stat_class, exceptions)))
def case_register_class_fail(args, cls, exception):
    return args, cls, exception


stat_funcs = ["accuracy", "true_positive_rate"]
stat_params = [["tp", "tn", "fp", "fn"], ["tp", "fn"]]


@parametrize("name, params", list(zip(stat_funcs, stat_params)))
def case_get_param(name, params):
    return name, params


stat_funcs = ["arbitrary"]


@parametrize("name", stat_funcs)
def case_get_param_fail(name):
    return name
