"""
Test functionality for computing_categorical_metrics.py
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from pytest_cases import parametrize
import pandas as pd


expected_dfs = [
    pd.DataFrame(
        {
            "band": {0: "1"},
            "fn": {0: 7315.0},
            "fp": {0: 5943.0},
            "tn": {0: 4845.0},
            "tp": {0: 22248.0},
            "accuracy": {0: 0.671433173899036},
            "critical_success_index": {0: 0.6265983214104658},
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
            "critical_success_index": {
                0: 0.2740790057700843,
                1: 0.44029443838604143,
            },
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
]

metrics_input = [
    "all",
    "all",
    "accuracy",
    ["accuracy", "critical_success_index", "f_score"],
]
positive_categories_input = [(1, 3), 2, 2, 2]
negative_categories_input = [2, 1, 1, 1, 1]


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


@parametrize(
    "crosstab_df, positive_categories, negative_categories",
    list(
        zip(
            input_dfs,
            positive_categories_input,
            negative_categories_input,
        )
    ),
)
def case_compute_categorical_metrics_fail(
    crosstab_df, positive_categories, negative_categories
):
    return crosstab_df, positive_categories, negative_categories
