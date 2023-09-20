"""
Test functionality for gval/subsampling.py
"""

# __all__ = ['*']

import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import rioxarray as rxr
from pytest_cases import parametrize

from tests.conftest import _load_xarray


candidate_map_fns = [
    _load_xarray(
        "candidate_continuous_1.tif", mask_and_scale=True, band_as_variable=True
    ),
    _load_xarray(
        "candidate_map_multiband_two_class_categorical.tif", mask_and_scale=True
    ),
]

benchmark_map_fns = [
    _load_xarray(
        "benchmark_continuous_1.tif", mask_and_scale=True, band_as_variable=True
    ),
    _load_xarray(
        "benchmark_map_multiband_two_class_categorical.tif", mask_and_scale=True
    ),
]

subsampling_dataframes = [
    gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (-97.72375, 29.56328),
                    (-97.72304, 29.55858),
                    (-97.71574, 29.55874),
                    (-97.72038, 29.56409),
                    (-97.72375, 29.56328),
                ]
            ),
            Polygon(
                [
                    (-97.71604, 29.55635),
                    (-97.71587, 29.55196),
                    (-97.71131, 29.55277),
                    (-97.71180, 29.55519),
                    (-97.71267, 29.55619),
                    (-97.71604, 29.55635),
                ]
            ),
        ],
        crs="EPSG:4326",
    ),
    gpd.GeoDataFrame(
        geometry=[
            Polygon(
                [
                    (-97.56735, 30.07450),
                    (-97.51800, 29.96872),
                    (-97.29514, 29.97237),
                    (-97.36734, 30.04412),
                    (-97.44813, 30.06807),
                    (-97.56735, 30.07450),
                ]
            ),
            Polygon(
                [
                    (-97.21658, 30.07011),
                    (-97.20986, 29.95981),
                    (-97.11101, 29.93847),
                    (-97.12969, 30.07781),
                    (-97.21658, 30.07011),
                ]
            ),
        ],
        crs="EPSG:4326",
    ),
]


create_dataframe_options = [
    {"subsampling_type": ["include", "include"], "subsampling_weights": [1, 2]},
    {"subsampling_type": ["exclude", "exclude"], "subsampling_weights": None},
]

expected_dfs = [
    pd.DataFrame(
        {
            "band": {0: "1", 1: "1", 2: "2", 3: "2"},
            "subsample": {0: "1", 1: "2", 2: "1", 3: "2"},
            "fn": {0: 201956.0, 1: 182394.0, 2: 68241.0, 3: 58642.0},
            "fp": {0: 761231.0, 1: 397538.0, 2: 43645.0, 3: 65969.0},
            "tn": {0: 762262.0, 1: 398349.0, 2: 1479848.0, 3: 729918.0},
            "tp": {0: 201946.0, 1: 181309.0, 2: 335661.0, 3: 305061.0},
            "accuracy": {
                0: 0.5002648652715194,
                1: 0.4998818547935046,
                2: 0.9419496263090856,
                3: 0.892538742141619,
            },
            "balanced_accuracy": {
                0: 0.5001629939598473,
                1: 0.4995089463871429,
                2: 0.9011988338571475,
                3: 0.8779383258349349,
            },
            "critical_success_index": {
                0: 0.17332441875734358,
                1: 0.23817555806899524,
                2: 0.7500016758016477,
                3: 0.7099857565771099,
            },
            "equitable_threat_score": {
                0: 0.00010804127698299228,
                1: -0.0004229210476480578,
                2: 0.6960116888476541,
                3: 0.6022620032041349,
            },
            "f_score": {
                0: 0.2954415948163932,
                1: 0.3847201739960745,
                2: 0.8571439515428851,
                3: 0.8303996145538584,
            },
            "false_discovery_rate": {
                0: 0.790333448576949,
                1: 0.6867756073712051,
                2: 0.11506540893104776,
                3: 0.17779963884322023,
            },
            "false_negative_rate": {
                0: 0.5000123792405088,
                1: 0.5014916016639951,
                2: 0.16895435031269962,
                3: 0.161235953511519,
            },
            "false_omission_rate": {
                0: 0.20945055993561623,
                1: 0.3140700791916562,
                2: 0.04408079897215212,
                3: 0.07436593283960637,
            },
            "false_positive_rate": {
                0: 0.4996616328397964,
                1: 0.4994905055617192,
                2: 0.028647981973005457,
                3: 0.08288739481861118,
            },
            "fowlkes_mallows_index": {
                0: 0.3237756633826248,
                1: 0.39515185725129576,
                2: 0.857566931595233,
                3: 0.8304409081615326,
            },
            "matthews_correlation_coefficient": {
                0: 0.0002653499871125589,
                1: -0.0009113478392335262,
                2: 0.821400707126061,
                3: 0.7518447869606043,
            },
            "negative_likelihood_ratio": {
                0: 0.9993484650766541,
                1: 1.0019622149761942,
                2: 0.17393730303446414,
                3: 0.1758082405591071,
            },
            "negative_predictive_value": {
                0: 0.7905494400643838,
                1: 0.6859299208083438,
                2: 0.9559192010278479,
                3: 0.9256340671603936,
            },
            "overall_bias": {
                0: 2.3846799471158846,
                1: 1.5915376007346653,
                2: 0.9391040400889324,
                3: 1.0201455583264367,
            },
            "positive_likelihood_ratio": {
                0: 1.0006524173526032,
                1: 0.9980337819942947,
                2: 29.008872264384337,
                3: 10.119319690575539,
            },
            "positive_predictive_value": {
                0: 0.20966655142305102,
                1: 0.3132243926287948,
                2: 0.8849345910689522,
                3: 0.8222003611567797,
            },
            "prevalence": {
                0: 0.2095584973500502,
                1: 0.3136479272846437,
                2: 0.2095584973500502,
                3: 0.3136479272846437,
            },
            "prevalence_threshold": {
                0: 0.499918474423107,
                1: 0.500246019173881,
                2: 0.15659282786558001,
                3: 0.23917220687599133,
            },
            "true_negative_rate": {
                0: 0.5003383671602035,
                1: 0.5005094944382809,
                2: 0.9713520180269946,
                3: 0.9171126051813888,
            },
            "true_positive_rate": {
                0: 0.49998762075949116,
                1: 0.4985083983360049,
                2: 0.8310456496873004,
                3: 0.838764046488481,
            },
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "1", 1: "2"},
            "band": {0: "averaged", 1: "averaged"},
            "fn": {0: 270197.0, 1: 241036.0},
            "fp": {0: 804876.0, 1: 463507.0},
            "tn": {0: 2242110.0, 1: 1128267.0},
            "tp": {0: 537607.0, 1: 486370.0},
            "accuracy": {0: 0.7211072457903025, 1: 0.6962102984675618},
            "balanced_accuracy": {0: 0.7006809139084974, 1: 0.6887236361110389},
            "critical_success_index": {0: 0.3333624773668676, 1: 0.4084009495236008},
            "equitable_threat_score": {0: 0.19249486119104803, 1: 0.21102574698031598},
            "f_score": {0: 0.5000327863210818, 1: 0.5799498355375926},
            "false_discovery_rate": {0: 0.599542787506434, 1: 0.48796528392623467},
            "false_negative_rate": {0: 0.33448336477660423, 1: 0.33136377758775704},
            "false_omission_rate": {0: 0.10754935603013485, 1: 0.1760282421056552},
            "false_positive_rate": {0: 0.26415480740640096, 1: 0.29118895019016516},
            "fowlkes_mallows_index": {0: 0.5162469724944239, 1: 0.5851196102503213},
            "matthews_correlation_coefficient": {
                0: 0.3428732020034822,
                1: 0.35612459483756775,
            },
            "negative_likelihood_ratio": {
                0: 0.4545567031533717,
                1: 0.46749239825854555,
            },
            "negative_predictive_value": {0: 0.8924506439698652, 1: 0.8239717578943447},
            "overall_bias": {0: 1.6618919936024086, 1: 1.3058415795305511},
            "positive_likelihood_ratio": {0: 2.5194189791878423, 1: 2.2962280058208955},
            "positive_predictive_value": {
                0: 0.40045721249356603,
                1: 0.5120347160737654,
            },
            "prevalence": {0: 0.2095584973500502, 1: 0.3136479272846437},
            "prevalence_threshold": {0: 0.38650811907632576, 1: 0.39756199260694686},
            "true_negative_rate": {0: 0.735845192593599, 1: 0.7088110498098348},
            "true_positive_rate": {0: 0.6655166352233958, 1: 0.668636222412243},
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "averaged", 1: "averaged"},
            "band": {0: "1", 1: "2"},
            "fn": {0: 384350.0, 1: 126883.0},
            "fp": {0: 1158769.0, 1: 109614.0},
            "tn": {0: 1160611.0, 1: 2209766.0},
            "tp": {0: 383255.0, 1: 640722.0},
            "accuracy": {0: 0.5001209918415541, 1: 0.9233890025380752},
            "balanced_accuracy": {0: 0.4998419157037689, 1: 0.8937213503088111},
            "critical_success_index": {0: 0.19895150162948627, 1: 0.7304014162939927},
            "equitable_threat_score": {
                0: -0.00011818049725140345,
                1: 0.6575691308727142,
            },
            "f_score": {0: 0.33187581208930095, 1: 0.8441988193216996},
            "false_discovery_rate": {0: 0.7514597697571503, 1: 0.1460865532241556},
            "false_negative_rate": {0: 0.5007132574696621, 1: 0.16529725575002768},
            "false_omission_rate": {0: 0.2487765063325223, 1: 0.054301266471772185},
            "false_positive_rate": {0: 0.4996029111228001, 1: 0.04726004363235002},
            "fowlkes_mallows_index": {0: 0.35226813927134054, 1: 0.8442534556492796},
            "matthews_correlation_coefficient": {
                0: -0.00027331863951123524,
                1: 0.7935041113348965,
            },
            "negative_likelihood_ratio": {0: 1.000631835395309, 1: 0.17349671822333187},
            "negative_predictive_value": {0: 0.7512234936674776, 1: 0.9456987335282279},
            "overall_bias": {0: 2.0088769614580415, 1: 0.9775027520664925},
            "positive_likelihood_ratio": {0: 0.9993671602278065, 1: 17.661912264478087},
            "positive_predictive_value": {
                0: 0.24854023024284966,
                1: 0.8539134467758445,
            },
            "prevalence": {0: 0.2486584806858472, 1: 0.2486584806858472},
            "prevalence_threshold": {0: 0.5000791300118148, 1: 0.19221129858834987},
            "true_negative_rate": {0: 0.5003970888771999, 1: 0.95273995636765},
            "true_positive_rate": {0: 0.4992867425303379, 1: 0.8347027442499723},
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "1", 1: "1", 2: "2", 3: "2"},
            "band": {0: "1", 1: "2", 2: "1", 3: "2"},
            "mean_percentage_error": {
                0: 0.1259589046239853,
                1: -0.11186813563108444,
                2: 0.1671745926141739,
                3: -0.1432301551103592,
            },
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "1", 1: "2"},
            "band": {0: "averaged", 1: "averaged"},
            "mean_percentage_error": {0: 0.007045384496450424, 1: 0.011972218751907349},
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "averaged", 1: "averaged"},
            "band": {0: "1", 1: "2"},
            "mean_percentage_error": {0: 0.1465667486190796, 1: -0.12754914537072182},
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "averaged", 1: "averaged"},
            "band": {0: "1", 1: "2"},
            "mean_percentage_error": {0: 0.08397260308265686, 1: -0.037289378543694816},
        }
    ),
]


@parametrize("df, args", list(zip(subsampling_dataframes, create_dataframe_options)))
def case_create_sampling_dataframes(df, args):
    return df, args


create_dataframe_options_fail = [
    {
        "geometry": [Point(1, 1), Point(1, 1)],
        "subsampling_type": ["include", "include"],
        "subsampling_weights": [1, 2],
    }
]


@parametrize(
    "df, args", list(zip(subsampling_dataframes[0:1], create_dataframe_options_fail))
)
def case_create_sampling_dataframes_fail(df, args):
    return df, args


expected_lengths = [2, 2]
subsample_types = ["exclude", "include"]
sample_percents = [[6.826683809589588], [6.826683809589588, 4.107072429734824]]


@parametrize(
    "candidate, benchmark, subsample_df, subsample_type, expected_length, sample_percent",
    list(
        zip(
            [candidate_map_fns[1], candidate_map_fns[1]],
            [benchmark_map_fns[1], benchmark_map_fns[1]],
            [subsampling_dataframes[1], subsampling_dataframes[1]],
            subsample_types,
            expected_lengths,
            sample_percents,
        )
    ),
)
def case_subsampling(
    candidate, benchmark, subsample_df, subsample_type, expected_length, sample_percent
):
    candidate = candidate.sel({"band": 1}) if expected_length == 1 else candidate
    benchmark = benchmark.sel({"band": 1}) if expected_length == 1 else benchmark
    return (
        candidate,
        benchmark,
        subsample_df,
        subsample_type,
        expected_length,
        sample_percent,
    )


exceptions = [ValueError, rxr.exceptions.NoDataInBounds]


@parametrize(
    "candidate, benchmark, subsample_df, exception",
    list(
        zip(
            [candidate_map_fns[0], candidate_map_fns[0]],
            [benchmark_map_fns[0], benchmark_map_fns[0]],
            subsampling_dataframes,
            exceptions,
        )
    ),
)
def case_subsampling_fail(candidate, benchmark, subsample_df, exception):
    return (
        candidate,
        benchmark,
        subsample_df,
        exception,
    )


sample_average_types = ["full-detail", "band", "subsample"]


@parametrize(
    "candidate, benchmark, subsample_df, expected_df, sampling_average",
    list(
        zip(
            [candidate_map_fns[1], candidate_map_fns[1], candidate_map_fns[1]],
            [benchmark_map_fns[1], benchmark_map_fns[1], benchmark_map_fns[1]],
            [
                subsampling_dataframes[1],
                subsampling_dataframes[1],
                subsampling_dataframes[1],
            ],
            [expected_dfs[0], expected_dfs[1], expected_dfs[2]],
            sample_average_types,
        )
    ),
)
def case_categorical_subsample(
    candidate, benchmark, subsample_df, expected_df, sampling_average
):
    return (
        candidate,
        benchmark,
        subsample_df,
        expected_df,
        sampling_average,
    )


sample_average_types = ["full-detail", "band", "subsample", "weighted"]


@parametrize(
    "candidate, benchmark, subsample_df, expected_df, sampling_average",
    list(
        zip(
            [
                candidate_map_fns[0],
                candidate_map_fns[0],
                candidate_map_fns[0],
                candidate_map_fns[0],
            ],
            [
                benchmark_map_fns[0],
                benchmark_map_fns[0],
                benchmark_map_fns[0],
                benchmark_map_fns[0],
            ],
            [
                subsampling_dataframes[0],
                subsampling_dataframes[0],
                subsampling_dataframes[0],
                subsampling_dataframes[0],
            ],
            [expected_dfs[3], expected_dfs[4], expected_dfs[5], expected_dfs[6]],
            sample_average_types,
        )
    ),
)
def case_continuous_subsample(
    candidate, benchmark, subsample_df, expected_df, sampling_average
):
    return (
        candidate,
        benchmark,
        subsample_df,
        expected_df,
        sampling_average,
    )
