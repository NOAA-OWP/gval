"""
Categorical Statistics Functions
"""

import math
from numbers import Number


def true_positive_rate(tp: Number, fn: Number) -> float:
    """
    Computes true positive rate, AKA sensitivity, recall, hit rate

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    True positive rate from 0 to 1

    References
    ----------
    .. [1] [Sensitivity and Specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    """
    return tp / (tp + fn)


def true_negative_rate(tn: Number, fp: Number) -> float:
    """
    Computes true negative rate, AKA specificity, selectivity

    Parameters
    ----------
    tn: Number
        Count reflecting true negative
    fp: Number
        Count reflecting false positive

    Returns
    -------
    True negative rate from 0 to 1

    References
    ----------
    .. [1] [Sensitivity and Specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    """
    return tn / (tn + fp)


def positive_predictive_value(tp: Number, fp: Number) -> float:
    """
    Computes positive predictive value AKA precision

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    fp: Number
        Count reflecting false positive

    Returns
    -------
    Positive predictive value from 0 to 1

    References
    ----------
    .. [1] [Sensitivity and Specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    """
    return tp / (tp + fp)


def negative_predictive_value(tn: Number, fn: Number) -> float:
    """
    Computes negative predictive value

    Parameters
    ----------
    tn: Number
        Count reflecting true negative
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Negative predictive value from 0 to 1

    References
    ----------
    .. [1] [Sensitivity and Specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
    """
    return tn / (tn + fn)


def false_negative_rate(tp: Number, fn: Number) -> float:
    """
    Computes false negative rate

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    False negative rate from 0 to 1

    References
    ----------
    .. [1] [Type I and Type II Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates)
    """
    return fn / (fn + tp)


def false_positive_rate(tn: Number, fp: Number) -> float:
    """
    Computes false positive rate AKA fall-out

    Parameters
    ----------
    tn: Number
        Count reflecting true negative
    fp: Number
        Count reflecting false positive

    Returns
    -------
    False positive rate from 0 to 1

    References
    ----------
    .. [1] [Type I and Type II Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates)
    """
    return fp / (fp + tn)


def false_discovery_rate(tp: Number, fp: Number) -> float:
    """
    Computes false discovery rate

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    fp: Number
        Count reflecting false positive

    Returns
    -------
    False discovery rate from 0 to 1

    References
    ----------
    .. [1] [False Discovery Rate](https://en.wikipedia.org/wiki/False_discovery_rate)
    """
    return fp / (fp + tp)


def false_omission_rate(tn: Number, fn: Number) -> float:
    """
    Computes false omission rate

    Parameters
    ----------
    tn: Number
        Count reflecting true negative
    fn: Number
        Count reflecting false negative

    Returns
    -------
    False omission rate from 0 to 1

    References
    ----------
    .. [1] [Positive and Negative Predictive Values](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values)
    """
    return fn / (fn + tn)


def positive_likelihood_ratio(tp: Number, tn: Number, fp: Number, fn: Number) -> float:
    """
    Computes positive likelihood ratio

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    tn: Number
        Count reflecting true negative
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Positive likelihood rate from 1 to infinity

    References
    ----------
    .. [1] [Likelihood Ratios](https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio)
    """
    return (tp / (tp + fn)) / (fp / (fp + tn))


def negative_likelihood_ratio(tp: Number, tn: Number, fp: Number, fn: Number) -> float:
    """
    Computes negative likelihood ratio

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    tn: Number
        Count reflecting true negative
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Negative likelihood from 1 to infinity

    References
    ----------
    .. [1] [Likelihood Ratios](https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio)
    """
    return (fn / (fn + tp)) / (tn / (tn + fp))


def prevalence_threshold(tp: Number, tn: Number, fp: Number, fn: Number) -> float:
    """
    Computes prevalence threshold

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    tn: Number
        Count reflecting true negative
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Prevalence threshold from 0 to 1

    References
    ----------
    .. [1] [Prevalence Threshold](https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Prevalence_threshold)
    """
    return math.sqrt(fp / (fp + tn)) / (
        math.sqrt(tp / (tp + fn)) + math.sqrt(fp / (fp + tn))
    )


def critical_success_index(tp: Number, fp: Number, fn: Number) -> float:
    """
    Computes critical success index

    https://www.weather.gov/media/erh/ta2004-03.pdf

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Critical success index from 0 to 1

    References
    ----------
    .. [1] [Critical Success Index](https://www.weather.gov/media/erh/ta2004-03.pdf)
    """
    return tp / (tp + fn + fp)


def prevalence(tp: Number, tn: Number, fp: Number, fn: Number) -> float:
    """
    Computes prevalence

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    tn: Number
        Count reflecting true negative
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Prevalence from 0 to 1

    References
    ----------
    .. [1] [Prevalence](https://en.wikipedia.org/wiki/Prevalence)
    """
    return (tp + fp) / (tp + fp + tn + fn)


def accuracy(tp: Number, tn: Number, fp: Number, fn: Number) -> float:
    """
    Computes accuracy

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    tn: Number
        Count reflecting true negative
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Accuracy from 0 to 1

    References
    ----------
    .. [1] [Accuracy and Precision](https://en.wikipedia.org/wiki/Accuracy_and_precision)
    """
    return (tp + tn) / (tp + fp + tn + fn)


def f_score(tp: Number, fp: Number, fn: Number) -> float:
    """
    Computes F-score AKA harmonic mean of precision and sensitivity

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    F-score from 0 to 1

    References
    ----------
    .. [1] [F-score](https://en.wikipedia.org/wiki/F-score)
    """
    return 2 * tp / (2 * tp + fp + fn)


def matthews_correlation_coefficient(
    tp: Number, tn: Number, fp: Number, fn: Number
) -> float:
    """
    Computes matthews correlation coefficient, accounting for accuracy and
    precision of both true positives and true negatives AKA Phi Coefficient

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    tn: Number
        Count reflecting true negative
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Correlation coefficient from -1 to 1

    References
    ----------
    .. [1] [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Phi_coefficient)
    """
    return (tp * tn - fp * fn) / math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )


def fowlkes_mallows_index(tp: Number, fp: Number, fn: Number) -> float:
    """
    Computes Fowlkes-Mallows index

    Parameters
    ----------
    tp: Number
        Count reflecting true positive
    fp: Number
        Count reflecting false positive
    fn: Number
        Count reflecting false negative

    Returns
    -------
    Correlation coefficient from -1 to 1

    References
    ----------
    .. [1] [Fowlkes-Mallows Index](https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index)
    """
    return math.sqrt((tp / (tp + fp)) * (tp / (tp + fn)))
