"""
Categorical Statistics Functions
"""

import math
from numbers import Number


class CategoricalStatistics:
    @staticmethod
    def true_positive_rate(tp: Number, fn: Number) -> float:  # pragma: no cover
        """
        Computes true positive rate, AKA sensitivity, recall, hit rate

        https://en.wikipedia.org/wiki/Sensitivity_and_specificity

        Parameters
        ----------
        tp: Number
            Count reflecting true positive
        fn: Number
            Count reflecting false negative

        Returns
        -------
        True positive rate from 0 to 1

        """
        return tp / (tp + fn)

    @staticmethod
    def true_negative_rate(tn: Number, fp: Number) -> float:  # pragma: no cover
        """
        Computes true negative rate, AKA specificity, selectivity

        https://en.wikipedia.org/wiki/Sensitivity_and_specificity

        Parameters
        ----------
        tn: Number
            Count reflecting true negative
        fp: Number
            Count reflecting false positive

        Returns
        -------
        True negative rate from 0 to 1
        """
        return tn / (tn + fp)

    @staticmethod
    def positive_predictive_value(tp: Number, fp: Number) -> float:  # pragma: no cover
        """
        Computes positive predictive value AKA precision

        https://en.wikipedia.org/wiki/Sensitivity_and_specificity

        Parameters
        ----------
        tp: Number
            Count reflecting true positive
        fp: Number
            Count reflecting false positive

        Returns
        -------
        Positive predictive value from 0 to 1
        """
        return tp / (tp + fp)

    @staticmethod
    def negative_predictive_value(tn: Number, fn: Number) -> float:  # pragma: no cover
        """
        Computes negative predictive value

        https://en.wikipedia.org/wiki/Sensitivity_and_specificity

        Parameters
        ----------
        tn: Number
            Count reflecting true negative
        fn: Number
            Count reflecting false negative

        Returns
        -------
        Negative predictive value from 0 to 1
        """
        return tn / (tn + fn)

    @staticmethod
    def false_negative_rate(tp: Number, fn: Number) -> float:  # pragma: no cover
        """
        Computes false negative rate

        https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates

        Parameters
        ----------
        tp: Number
            Count reflecting true positive
        fn: Number
            Count reflecting false negative

        Returns
        -------
        False negative rate from 0 to 1
        """
        return fn / (fn + tp)

    @staticmethod
    def false_positive_rate(tn: Number, fp: Number) -> float:  # pragma: no cover
        """
        Computes false positive rate AKA fall-out

        https://en.wikipedia.org/wiki/Type_I_and_type_II_errors#False_positive_and_false_negative_rates

        Parameters
        ----------
        tn: Number
            Count reflecting true negative
        fp: Number
            Count reflecting false positive

        Returns
        -------
        False positive rate from 0 to 1
        """
        return fp / (fp + tn)

    @staticmethod
    def false_discovery_rate(tp: Number, fp: Number) -> float:  # pragma: no cover
        """
        Computes false discovery rate

        https://en.wikipedia.org/wiki/False_discovery_rate

        Parameters
        ----------
        tp: Number
            Count reflecting true positive
        fp: Number
            Count reflecting false positive

        Returns
        -------
        False discovery rate from 0 to 1
        """
        return fp / (fp + tp)

    @staticmethod
    def false_omission_rate(tn: Number, fn: Number) -> float:  # pragma: no cover
        """
        Computes false omission rate

        https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values

        Parameters
        ----------
        tn: Number
            Count reflecting true negative
        fn: Number
            Count reflecting false negative

        Returns
        -------
        False omission rate from 0 to 1
        """
        return fn / (fn + tn)

    @staticmethod
    def positive_likelihood_ratio(
        tp: Number, tn: Number, fp: Number, fn: Number
    ) -> float:  # pragma: no cover
        """
        Computes positive likelihood ratio

        https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio

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
        """
        return (tp / (tp + fn)) / (fp / (fp + tn))

    @staticmethod
    def negative_likelihood_ratio(
        tp: Number, tn: Number, fp: Number, fn: Number
    ) -> float:  # pragma: no cover
        """
        Computes negative likelihood ratio

        https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#negative_likelihood_ratio

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
        """
        return (fn / (fn + tp)) / (tn / (tn + fp))

    @staticmethod
    def prevalence_threshold(
        tp: Number, tn: Number, fp: Number, fn: Number
    ) -> float:  # pragma: no cover
        """
        Computes prevalence threshold

        https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Prevalence_threshold

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
        """
        return math.sqrt(fp / (fp + tn)) / (
            math.sqrt(tp / (tp + fn)) + math.sqrt(fp / (fp + tn))
        )

    @staticmethod
    def critical_success_index(
        tp: Number, fp: Number, fn: Number
    ) -> float:  # pragma: no cover
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
        """
        return tp / (tp + fn + fp)

    @staticmethod
    def prevalence(
        tp: Number, tn: Number, fp: Number, fn: Number
    ) -> float:  # pragma: no cover
        """
        Computes prevalence

        https://en.wikipedia.org/wiki/Prevalence

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
        """
        return (tp + fp) / (tp + fp + tn + fn)

    @staticmethod
    def accuracy(
        tp: Number, tn: Number, fp: Number, fn: Number
    ) -> float:  # pragma: no cover
        """
        Computes accuracy

        https://en.wikipedia.org/wiki/Accuracy_and_precision

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
        """
        return (tp + tn) / (tp + fp + tn + fn)

    @staticmethod
    def f_score(tp: Number, fp: Number, fn: Number) -> float:  # pragma: no cover
        """
        Computes F-score AKA harmonic mean of precision and sensitivity

        https://en.wikipedia.org/wiki/F-score

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
        """
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def matthews_correlation_coefficient(
        tp: Number, tn: Number, fp: Number, fn: Number
    ) -> float:  # pragma: no cover
        """
        Computes matthews correlation coefficient, accounting for accuracy and
        precision of both true positives and true negatives AKA Phi Coefficient

        https://en.wikipedia.org/wiki/Phi_coefficient

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

        """
        return (tp * tn - fp * fn) / math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )

    @staticmethod
    def fowlkes_mallows_index(
        tp: Number, fp: Number, fn: Number
    ) -> float:  # pragma: no cover
        """
        Computes Fowlkes-Mallows index

        https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index

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

        """
        return math.sqrt((tp / (tp + fp)) * (tp / (tp + fn)))
