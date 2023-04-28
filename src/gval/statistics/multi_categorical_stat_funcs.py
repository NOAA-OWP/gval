"""
Multi-class Categorical Metric Functions

These work on a confusion matrix schema.

    average{‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’
    If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:

    'binary':
    Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.

    'micro':
    Calculate metrics globally by counting the total true positives, false negatives and false positives.

    'macro':
    Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

    'weighted':
    Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance.

    'samples':
    Calculate metrics for each instance, and find their average (only meaningful for multilabel classification).

"""

import numpy as np
import pandas as pd

from gval.utils.schemas import Pivoted_crosstab_df


def _compute_totals(Pivoted_crosstab_df):
    """
    Computes the total number of samples in each category.

    Parameters
    ----------
    Pivoted_crosstab_df : DataFrame[Pivot_crosstab_df]
        A pivoted crosstab DataFrame.

    Returns
    -------
    Tuple[Series[float], Series[float], float, float]
        A tuple of the per category candidate totals, per category benchmark totals, diagnol or correctly predicted totals, and overall total. Totals represent the frequency of occurrence of each category in spatial units or counts.
    """
    # sum totals
    candidate_totals = Pivoted_crosstab_df.sum(axis=0)
    benchmark_totals = Pivoted_crosstab_df.sum(axis=1)

    # get total number of counts
    total = candidate_totals.sum()

    # correct values
    overlapping_values = list(
        set(Pivoted_crosstab_df.index) & set(Pivoted_crosstab_df.columns)
    )
    correct_total = np.diag(
        Pivoted_crosstab_df.loc[overlapping_values, overlapping_values]
    ).sum()

    # rename series
    candidate_totals.name = "candidate_totals"
    benchmark_totals.name = "benchmark_totals"

    return candidate_totals, benchmark_totals, correct_total, total


def matthews_correlation_coefficient(Pivoted_crosstab_df):
    """
    Computes the Matthews Correlation Coefficient.

    Parameters
    ----------
    Pivoted_crosstab_df : DataFrame[Pivot_crosstab_df]
        A pivoted crosstab DataFrame.

    Returns
    -------
    float
        The Matthews Correlation Coefficient.
    """
    candidate_totals, benchmark_totals, correct_total, overall_total = _compute_totals(
        Pivoted_crosstab_df
    )

    # compute matthews correlation coefficient on confusion matrix
    mcc = (
        overall_total * correct_total - np.dot(candidate_totals * benchmark_totals)
    ) / np.sqrt(
        overall_total**2
        - np.sum(candidate_totals**2)
        * np.sqrt(overall_total**2 - np.sum(benchmark_totals**2))
    )

    return mcc


def kappa_coefficient(Pivoted_crosstab_df):
    """
    Computes the Kappa Coefficient.

    Parameters
    ----------
    Pivoted_crosstab_df : DataFrame[Pivot_crosstab_df]
        A pivoted crosstab DataFrame.

    Returns
    -------
    float
        The Kappa Coefficient.
    """

    candidate_totals, benchmark_totals, correct_total, overall_total = _compute_totals(
        Pivoted_crosstab_df
    )

    # compute dot product of totals
    dot_product = np.dot(candidate_totals * benchmark_totals)

    # compute kappa on confusion matrix
    return (correct_total - dot_product) / (overall_total - dot_product)


def accuracy(Pivoted_crosstab_df):
    """
    Computes the accuracy.

    Parameters
    ----------
    Pivoted_crosstab_df : DataFrame[Pivot_crosstab_df]
        A pivoted crosstab DataFrame.

    Returns
    -------
    float
        The accuracy.
    """

    candidate_totals, benchmark_totals, correct_total, overall_total = _compute_totals(
        Pivoted_crosstab_df
    )

    # compute accuracy on confusion matrix
    return correct_total / overall_total
