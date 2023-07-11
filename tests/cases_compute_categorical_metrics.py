"""
Test functionality for computing_categorical_metrics.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

import numpy as np
from pytest_cases import parametrize
import pandas as pd
from gval.utils.schemas import Metrics_df


expected_dfs = [
    pd.DataFrame(
        {
            "band": {0: "1"},
            "fn": {0: 7315.0},
            "fp": {0: 5943.0},
            "tn": {0: 4845.0},
            "tp": {0: 22248.0},
            "accuracy": {0: 0.671433173899036},
            "balanced_accuracy": {0: 0.6008362234427282},
            "critical_success_index": {0: 0.6265983214104658},
            "equitable_threat_score": {0: 0.10732415444447231},
            "f_score": {0: 0.7704401426741005},
            "false_discovery_rate": {0: 0.21081196126423327},
            "false_negative_rate": {0: 0.24743767547271928},
            "false_omission_rate": {0: 0.6015625},
            "false_positive_rate": {0: 0.5508898776418243},
            "fowlkes_mallows_index": {0: 0.7706576314551841},
            "matthews_correlation_coefficient": {0: 0.19452223907575214},
            "negative_likelihood_ratio": {0: 0.5509510099070579},
            "negative_predictive_value": {0: 0.3984375},
            "positive_likelihood_ratio": {0: 1.3660848657244327},
            "positive_predictive_value": {0: 0.7891880387357667},
            "prevalence": {0: 0.6986443954301008},
            "prevalence_threshold": {0: 0.46108525048654536},
            "true_negative_rate": {0: 0.44911012235817577},
            "true_positive_rate": {0: 0.7525623245272807},
        }
    ),
    pd.DataFrame(
        {
            "band": {0: "1", 1: "2"},
            "fn": {0: 1098.0, 1: 3689.0},
            "fp": {0: 5444.0, 1: 2470.0},
            "tn": {0: 9489.0, 1: 5200.0},
            "tp": {0: 2470.0, 1: 4845.0},
            "accuracy": {0: 0.646397492027458, 1: 0.6199086645272772},
            "balanced_accuracy": {0: 0.6638514325121567, 1: 0.6228475926801269},
            "critical_success_index": {
                0: 0.2740790057700843,
                1: 0.44029443838604143,
            },
            "equitable_threat_score": {0: 0.12607286705706663, 1: 0.1387798441467566},
            "f_score": {0: 0.43023863438425364, 1: 0.6113950406965739},
            "false_discovery_rate": {0: 0.6878948698508971, 1: 0.33766233766233766},
            "false_negative_rate": {0: 0.3077354260089686, 1: 0.43227091633466136},
            "false_omission_rate": {0: 0.10371209974497024, 1: 0.4150073124085949},
            "false_positive_rate": {0: 0.364561708966718, 1: 0.3220338983050847},
            "fowlkes_mallows_index": {
                0: 0.46482182066151334,
                1: 0.6132115084666983,
            },
            "matthews_correlation_coefficient": {
                0: 0.2613254543945788,
                1: 0.2465114118474816,
            },
            "negative_likelihood_ratio": {
                0: 0.4842884515325037,
                1: 0.6375996015936255,
            },
            "negative_predictive_value": {
                0: 0.8962879002550298,
                1: 0.5849926875914051,
            },
            "positive_likelihood_ratio": {
                0: 1.898895459847184,
                1: 1.762948207171315,
            },
            "positive_predictive_value": {
                0: 0.31210513014910285,
                1: 0.6623376623376623,
            },
            "prevalence": {0: 0.42776066158586024, 1: 0.4514317452480869},
            "prevalence_threshold": {0: 0.4205207112769604, 1: 0.42959744253992793},
            "true_negative_rate": {0: 0.635438291033282, 1: 0.6779661016949152},
            "true_positive_rate": {0: 0.6922645739910314, 1: 0.5677290836653387},
        }
    ),
    pd.DataFrame(
        {
            "band": {0: "1", 1: "2"},
            "fn": {0: 1098.0, 1: 3689.0},
            "fp": {0: 5444.0, 1: 2470.0},
            "tn": {0: 9489.0, 1: 5200.0},
            "tp": {0: 2470.0, 1: 4845.0},
            "accuracy": {0: 0.646397492027458, 1: 0.6199086645272772},
        }
    ),
    pd.DataFrame(
        {
            "band": {0: "1", 1: "2"},
            "fn": {0: 1098.0, 1: 3689.0},
            "fp": {0: 5444.0, 1: 2470.0},
            "tn": {0: 9489.0, 1: 5200.0},
            "tp": {0: 2470.0, 1: 4845.0},
            "accuracy": {0: 0.646397492027458, 1: 0.6199086645272772},
            "critical_success_index": {
                0: 0.2740790057700843,
                1: 0.44029443838604143,
            },
            "f_score": {0: 0.43023863438425364, 1: 0.6113950406965739},
        }
    ),
    pd.DataFrame(
        {
            "band": {0: "1", 1: "2"},
            "fn": {0: 1098.0, 1: 3689.0},
            "fp": {0: 5444.0, 1: 2470.0},
            "tn": {0: np.nan, 1: np.nan},
            "tp": {0: 2470.0, 1: 4845.0},
            "critical_success_index": {0: 0.274079, 1: 0.440294},
            "f_score": {0: 0.430239, 1: 0.611395},
            "false_discovery_rate": {0: 0.687895, 1: 0.337662},
            "false_negative_rate": {0: 0.307735, 1: 0.432271},
            "fowlkes_mallows_index": {0: 0.464822, 1: 0.613212},
            "positive_predictive_value": {0: 0.312105, 1: 0.662338},
            "true_positive_rate": {0: 0.692265, 1: 0.567729},
        }
    ),
]

input_dfs = [
    pd.DataFrame(
        {
            "band": {
                0: "1",
                1: "1",
                2: "1",
                3: "1",
                4: "1",
                5: "1",
                6: "1",
                7: "1",
                8: "1",
            },
            "candidate_values": {
                0: 1,
                1: 1,
                2: 1,
                3: 2,
                4: 2,
                5: 2,
                6: 3,
                7: 3,
                8: 3,
            },
            "benchmark_values": {
                0: 1,
                1: 2,
                2: 3,
                3: 1,
                4: 2,
                5: 3,
                6: 1,
                7: 2,
                8: 3,
            },
            "counts": {
                0: 9489,
                1: 1098,
                2: 5444,
                3: 2470,
                4: 4845,
                5: 4845,
                6: 2470,
                7: 4845,
                8: 4845,
            },
        }
    ),
    pd.DataFrame(
        {
            "band": {
                0: "1",
                1: "1",
                2: "1",
                3: "1",
                4: "2",
                5: "2",
                6: "2",
                7: "2",
            },
            "candidate_values": {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 2, 7: 2},
            "benchmark_values": {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 2},
            "counts": {
                0: 9489,
                1: 1098,
                2: 5444,
                3: 2470,
                4: 5200,
                5: 3689,
                6: 2470,
                7: 4845,
            },
        }
    ),
    pd.DataFrame(
        {
            "band": {
                0: "1",
                1: "1",
                2: "1",
                3: "1",
                4: "2",
                5: "2",
                6: "2",
                7: "2",
            },
            "candidate_values": {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 2, 7: 2},
            "benchmark_values": {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 2},
            "counts": {
                0: 9489,
                1: 1098,
                2: 5444,
                3: 2470,
                4: 5200,
                5: 3689,
                6: 2470,
                7: 4845,
            },
        }
    ),
    pd.DataFrame(
        {
            "band": {
                0: "1",
                1: "1",
                2: "1",
                3: "1",
                4: "2",
                5: "2",
                6: "2",
                7: "2",
            },
            "candidate_values": {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 2, 7: 2},
            "benchmark_values": {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 2},
            "counts": {
                0: 9489,
                1: 1098,
                2: 5444,
                3: 2470,
                4: 5200,
                5: 3689,
                6: 2470,
                7: 4845,
            },
        }
    ),
    pd.DataFrame(
        {
            "band": {
                0: "1",
                1: "1",
                2: "1",
                3: "1",
                4: "2",
                5: "2",
                6: "2",
                7: "2",
            },
            "candidate_values": {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 2, 7: 2},
            "benchmark_values": {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 2},
            "counts": {
                0: 9489,
                1: 1098,
                2: 5444,
                3: 2470,
                4: 5200,
                5: 3689,
                6: 2470,
                7: 4845,
            },
        }
    ),
]

metrics_input = [
    "all",
    "all",
    "accuracy",
    ["accuracy", "critical_success_index", "f_score"],
    "all",
]
positive_categories_input = [(1, 3), 2, 2, 2, 2]
negative_categories_input = [2, 1, 1, 1, None]


@parametrize(
    "crosstab_df, metrics, positive_categories, negative_categories, expected_df",
    list(
        zip(
            input_dfs,
            metrics_input,
            positive_categories_input,
            negative_categories_input,
            expected_dfs,
        )
    ),
)
def case_compute_categorical_metrics_success(
    crosstab_df, metrics, positive_categories, negative_categories, expected_df
):
    return crosstab_df, metrics, positive_categories, negative_categories, expected_df


input_dfs = [
    pd.DataFrame(
        {
            "band": {
                0: "1",
                1: "1",
                2: "1",
                3: "1",
                4: "1",
                5: "1",
                6: "1",
                7: "1",
                8: "1",
            },
            "candidate_values": {
                0: 1,
                1: 1,
                2: 1,
                3: 2,
                4: 2,
                5: 2,
                6: 3,
                7: 3,
                8: 3,
            },
            "benchmark_values": {
                0: 1,
                1: 2,
                2: 3,
                3: 1,
                4: 2,
                5: 3,
                6: 1,
                7: 2,
                8: 3,
            },
            "counts": {
                0: 9489,
                1: 1098,
                2: 5444,
                3: 2470,
                4: 4845,
                5: 4845,
                6: 2470,
                7: 4845,
                8: 4845,
            },
        }
    )
] * 3

negative_categories_input = [None, 4, (2, 3)]
positive_categories_input = [4, 3, (1, 2)]
exceptions = [ValueError, ValueError, ValueError]


@parametrize(
    "crosstab_df, positive_categories, negative_categories, exception",
    list(
        zip(input_dfs, positive_categories_input, negative_categories_input, exceptions)
    ),
)
def case_compute_categorical_metrics_fail(
    crosstab_df, positive_categories, negative_categories, exception
):
    return crosstab_df, positive_categories, negative_categories, exception


multi_class_input_df = [
    pd.DataFrame(
        {
            "band": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "candidate_values": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "benchmark_values": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "counts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
] * 4 + [
    pd.DataFrame(
        {
            "band": [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "candidate_values": [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "benchmark_values": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "counts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
]

metrics = [
    "critical_success_index",
    ["critical_success_index", "f_score"],
    ["critical_success_index", "f_score", "accuracy", "positive_predictive_value"],
    ["f_score", "false_omission_rate", "false_discovery_rate"],
    ["prevalence", "true_negative_rate", "true_positive_rate"],
]

positive_categories_input = [(1, 2), (1, 2, 3, 4), (1, 2, 3), 1, (1, 2, 3)]

negative_categories_input = [(3, 4), None, None, None, None]

averages_all = ["micro", "macro", "weighted", None, None]

weights_all = [None, None, (1, 2, 3), None, None]

expected_dfs = [
    Metrics_df.validate(
        pd.DataFrame(
            {
                "band": [1],
                "fn": 19,
                "fp": 7,
                "tn": 38,
                "tp": 14,
                "critical_success_index": 0.35,
            }
        )
    ),
    Metrics_df.validate(
        pd.DataFrame(
            {"band": [1], "critical_success_index": 0.180775, "f_score": 0.29776}
        )
    ),
    Metrics_df.validate(
        pd.DataFrame(
            {
                "band": [1],
                "critical_success_index": 0.34639,
                "f_score": 0.49563,
                "accuracy": 0.651515,
                "positive_predictive_value": 0.428346,
            }
        )
    ),
    Metrics_df.validate(
        pd.DataFrame(
            {
                "band": [1],
                "positive_categories": 1,
                "fn": 14,
                "fp": 5,
                "tn": np.nan,
                "tp": 1,
                "f_score": 0.095238,
                "false_discovery_rate": 0.833333,
            }
        )
    ),
    Metrics_df.validate(
        pd.DataFrame(
            {
                "band": [1, 2, 1, 2, 1, 2],
                "positive_categories": [1, 1, 2, 2, 3, 3],
                "fn": [11, 11, 10, 10, 9, 9],
                "fp": [5, 5, 10, 10, 15, 15],
                "tn": [28, 28, 20, 20, 12, 12],
                "tp": [1, 1, 5, 5, 9, 9],
                "prevalence": [
                    0.133333,
                    0.133333,
                    0.333333,
                    0.333333,
                    0.533333,
                    0.533333,
                ],
                "true_negative_rate": [
                    0.848485,
                    0.848485,
                    0.666667,
                    0.666667,
                    0.444444,
                    0.444444,
                ],
                "true_positive_rate": [
                    0.083333,
                    0.083333,
                    0.333333,
                    0.333333,
                    0.5,
                    0.5,
                ],
            }
        )
    ),
]


@parametrize(
    "crosstab_df, metrics, positive_categories, negative_categories, average, weights, expected_df",
    list(
        zip(
            multi_class_input_df,
            metrics,
            positive_categories_input,
            negative_categories_input,
            averages_all,
            weights_all,
            expected_dfs,
        )
    ),
)
def case_compute_multi_categorical_metrics_success(
    crosstab_df,
    metrics,
    positive_categories,
    negative_categories,
    average,
    weights,
    expected_df,
):
    return (
        crosstab_df,
        metrics,
        positive_categories,
        negative_categories,
        average,
        weights,
        expected_df,
    )


multi_class_input_df_fails = [
    pd.DataFrame(
        {
            "band": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "candidate_values": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "benchmark_values": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "counts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
] * 4

metrics = ["all", "all", "all", "all"]

positive_categories_input = [1, 2, [1, 2], [1, 2]]
negative_categories_input = [2, 1, 1, [3, 4]]

averages_all_fails = ["macro", "weighted", "weighted", "macro"]

weights_all_fails = [None, [2], [2], None]

exceptions = [ValueError] * 4


@parametrize(
    "crosstab_df, metrics, positive_categories, negative_categories, average, weights, exception",
    list(
        zip(
            multi_class_input_df_fails,
            metrics,
            positive_categories_input,
            negative_categories_input,
            averages_all_fails,
            weights_all_fails,
            exceptions,
        )
    ),
)
def case_compute_multi_categorical_metrics_fail(
    crosstab_df,
    metrics,
    positive_categories,
    negative_categories,
    average,
    weights,
    exception,
):
    return (
        crosstab_df,
        metrics,
        positive_categories,
        negative_categories,
        average,
        weights,
        exception,
    )
