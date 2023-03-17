"""
Categorical Statistics Functions
"""

import math


def true_positive_rate(tp: int, fn: int) -> float:
    """
    Computes true positive rate, AKA sensitivity, recall, hit rate

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    True positive rate from 0 to 1

    """
    return tp / (tp + fn)


def true_negative_rate(tn: int, fp: int) -> float:
    """
    Computes true negative rate, AKA specificty, selectivity

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Parameters
    ----------
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive

    Returns
    -------
    True negative rate from 0 to 1
    """
    return tn / (tn + fp)


def positive_predictive_value(tp: int, fp: int) -> float:
    """
    Computes positive predictive value AKA precision

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    fp: int
        Count reflecting false positive

    Returns
    -------
    Positive predictive value from 0 to 1
    """
    return tp / (tp + fp)


def negative_predictive_value(tn: int, fn: int) -> float:
    """
    Computes negative predictive value

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Negative predictive value from 0 to 1
    """
    return tn / (tn + fn)


def false_negative_rate(tp: int, fn: int) -> float:
    """
    Computes false negative rate

    https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    False negative rate from 0 to 1
    """
    return fn / (fn + tp)


def false_positive_rate(tn: int, fp: int) -> float:
    """
    Computes false positive rate AKA fall-out

    https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    False positive rate from 0 to 1
    """
    return fp / (fp + tn)


def false_discovery_rate(tp: int, fp: int) -> float:
    """
    Computes false discovery rate

    https://en.wikipedia.org/wiki/False_discovery_rate

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    False discovery rate from 0 to 1
    """
    return fp / (fp + tp)


def false_omission_rate(tn: int, fn: int) -> float:
    """
    Computes false omission rate

    https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values

    Parameters
    ----------
    tn: int
        Count reflecting true negative
    fn: int
        Count reflecting false negative

    Returns
    -------
    False omission rate from 0 to 1
    """
    return fn / (fn + tn)


def positive_likelihood_ratio(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Computes positive likelihood ratio

    https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Positive likelihood rate from 1 to infinity
    """
    return (tp / (tp + fn)) / (fp / (fp + tn))


def negative_likelihood_ratio(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Computes negative likelihood ratio

    https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#negative_likelihood_ratio

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Negative likelihood from 1 to infinity
    """
    return (fn / (fn + tp)) / (tn / (tn + fp))


def prevalence_threshold(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Computes prevalence threshold

    https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Prevalence_threshold

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Prevalence threshold from 0 to 1
    """
    return math.sqrt(fp / (fp + tn)) / (
        math.sqrt(tp / (tp + fn)) + math.sqrt(fp / (fp + tn))
    )


def critical_success_index(tp: int, fp: int, fn: int) -> float:
    """
    Computes critical success index

    https://www.weather.gov/media/erh/ta2004-03.pdf

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Critical success index from 0 to 1
    """
    return tp / (tp + fn + fp)


def prevalence(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Computes prevalence

    https://en.wikipedia.org/wiki/Prevalence

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Prevalence from 0 to 1
    """
    return (tp + fp) / (tp + fp + tn + fn)


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Computes accuracy

    https://en.wikipedia.org/wiki/Accuracy_and_precision

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Accuracy from 0 to 1
    """
    return (tp + tn) / (tp + fp + tn + fn)


def f_score(tp: int, fp: int, fn: int) -> float:
    """
    Computes F-score AKA harmonic mean of precision and sensitivity

    https://en.wikipedia.org/wiki/F-score

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    F-score from 0 to 1
    """
    return 2 * tp / (2 * tp + fp + fn)


def matthews_correlation_coefficient(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Computes matthews correlation coefficient, accounting for accuracy and
    precision of both true positives and true negatives AKA Phi Coefficient

    https://en.wikipedia.org/wiki/Phi_coefficient

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    tn: int
        Count reflecting true negative
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Correlation coefficient from -1 to 1

    """
    return (tp * tn - fp * fn) / math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )


def fowlkes_mallows_index(tp: int, fp: int, fn: int) -> float:
    """
    Computes Fowlkes-Mallows index

    https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index

    Parameters
    ----------
    tp: int
        Count reflecting true positive
    fp: int
        Count reflecting false positive
    fn: int
        Count reflecting false negative

    Returns
    -------
    Correlation coefficient from -1 to 1

    """
    return math.sqrt((tp / (tp + fp)) * (tp / (tp + fn)))
