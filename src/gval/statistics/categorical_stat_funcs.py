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
    float
        True positive rate from 0 to 1

    References
    ----------
    .. [1] `Sensitivity and Specificity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_
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
    float
        True negative rate from 0 to 1

    References
    ----------
    .. [1] `Sensitivity and Specificity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_
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
    float
        Positive predictive value from 0 to 1

    References
    ----------
    .. [1] `Sensitivity and Specificity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_
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
    float
        Negative predictive value from 0 to 1

    References
    ----------
    .. [1] `Sensitivity and Specificity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_
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
    float
        False negative rate from 0 to 1

    References
    ----------
    .. [1] `Type I and Type II Error <https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates>`_
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
    float
        False positive rate from 0 to 1

    References
    ----------
    .. [1] `Type I and Type II Errors <https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates>`_
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
    float
        False discovery rate from 0 to 1

    References
    ----------
    .. [1] `False Discovery Rate <https://en.wikipedia.org/wiki/False_discovery_rate>`_
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
    float
        False omission rate from 0 to 1

    References
    ----------
    .. [1] `Positive and Negative Predictive Values <https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values>`_
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
    float
        Positive likelihood rate from 1 to infinity

    References
    ----------
    .. [1] `Likelihood Ratios <https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio>`_
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
    float
        Negative likelihood from 1 to infinity

    References
    ----------
    .. [1] `Likelihood Ratios <https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio>`_
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
    float
        Prevalence threshold from 0 to 1

    References
    ----------
    .. [1] `Prevalence Threshold <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7540853/>`_
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
    float
        Critical success index from 0 to 1

    References
    ----------
    .. [1] `Critical Success Index <https://www.swpc.noaa.gov/sites/default/files/images/u30/Forecast%20Verification%20Glossary.pdf#page=4>`_
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
    float
        Prevalence from 0 to 1

    References
    ----------
    .. [1] `Prevalence <https://en.wikipedia.org/wiki/Prevalence>`_
    """
    return (tp + fn) / (tp + fp + tn + fn)


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
    float
        Accuracy from 0 to 1

    References
    ----------
    .. [1] `Accuracy and Precision <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_
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
    float
        F-score from 0 to 1

    References
    ----------
    .. [1] `F-score <https://en.wikipedia.org/wiki/F-score>`_
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
    float
        Correlation coefficient from -1 to 1

    References
    ----------
    .. [1] `Matthews Correlation Coefficient <https://en.wikipedia.org/wiki/Phi_coefficient>`_
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
    float
        Correlation coefficient from -1 to 1

    References
    ----------
    .. [1] `Fowlkes-Mallows Index <https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index>`_
    """
    return math.sqrt((tp / (tp + fp)) * (tp / (tp + fn)))


def equitable_threat_score(tp: Number, tn: Number, fp: Number, fn: Number) -> float:
    """
    Computes Equitable Threat Score (Gilbert Score)

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
    float
        Equitable threat score from -1/3 to 1

    References
    ----------
    .. [1] `Equitable Threat Score <https://resources.eumetrain.org/data/4/451/english/msg/ver_categ_forec/uos2/uos2_ko4.htm>`_

    """
    total_population = tp + tn + fp + fn
    a_ref = ((tp + fp) * (tp + fn)) / total_population
    return (tp - a_ref) / (tp - a_ref + fp + fn)


def balanced_accuracy(tp: Number, tn: Number, fp: Number, fn: Number) -> float:
    """
    Computes Balanced Accuracy

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
    float
        Balanced Accuracy from 0 to 1

    References
    ----------
    .. [1] `Balanced Accuracy <https://neptune.ai/blog/balanced-accuracy#Balanced%20Accuracy>`_
    """

    return ((tp / (tp + fn)) + (tn / (tn + fp))) / 2


def overall_bias(tp: Number, fp: Number, fn: Number) -> float:
    """
    Computes the degree of correspondence between the mean forecast and the mean observation.

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
    float
        Overall Bias

    References
    ----------
    .. [1] `Forecast Verification Glossary <https://www.swpc.noaa.gov/sites/default/files/images/u30/Forecast%20Verification%20Glossary.pdf>`_

    """

    return (tp + fp) / (tp + fn)
