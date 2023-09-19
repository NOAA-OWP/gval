"""
Test functionality for gval/subsampling.py
"""

# __all__ = ['*']

from shapely.geometry import Point
import pandas as pd
import rioxarray as rxr
from pytest_cases import parametrize

from tests.conftest import _load_xarray, _load_gpkg


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
    _load_gpkg("subsample_continuous_polygons.gpkg"),
    _load_gpkg("subsample_two-class_polygons.gpkg"),
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
            "fn": {0: 201953.0, 1: 182389.0, 2: 68239.0, 3: 58638.0},
            "fp": {0: 761242.0, 1: 397531.0, 2: 43646.0, 3: 65967.0},
            "tn": {0: 762262.0, 1: 398338.0, 2: 1479858.0, 3: 729902.0},
            "tp": {0: 201936.0, 1: 181301.0, 2: 335650.0, 3: 305052.0},
            "accuracy": {
                0: 0.5002601960264461,
                1: 0.4998788332460875,
                2: 0.9419500849074371,
                3: 0.8925410436208938,
            },
            "balanced_accuracy": {
                0: 0.5001568546160343,
                1: 0.49950560680624195,
                2: 0.9011983659441684,
                3: 0.8779412625210775,
            },
            "critical_success_index": {
                0: 0.17331613355064796,
                1: 0.23817130636175302,
                2: 0.7499972069223636,
                3: 0.7099895963524393,
            },
            "equitable_threat_score": {
                0: 0.0001039688574513486,
                1: -0.00042579383623114476,
                2: 0.6960082702999203,
                3: 0.6022678227722073,
            },
            "f_score": {
                0: 0.29542955831718565,
                1: 0.38471462735087353,
                2: 0.8571410330892446,
                3: 0.8304022408872084,
            },
            "false_discovery_rate": {
                0: 0.7903440485559263,
                1: 0.6867813113304033,
                2: 0.11507107905171686,
                3: 0.17779951970114738,
            },
            "false_negative_rate": {
                0: 0.5000210453862324,
                1: 0.5014957793725425,
                2: 0.16895483660114535,
                3: 0.1612307184690258,
            },
            "false_omission_rate": {
                0: 0.20944810026809374,
                1: 0.3140701224499636,
                2: 0.04407927926996823,
                3: 0.07436274634133969,
            },
            "false_positive_rate": {
                0: 0.49966524538169904,
                1: 0.49949300701497357,
                2: 0.02864843151051786,
                3: 0.08288675648881914,
            },
            "fowlkes_mallows_index": {
                0: 0.32376467292087746,
                1: 0.3951466035298691,
                2: 0.8575639333051723,
                3: 0.83044355987309,
            },
            "matthews_correlation_coefficient": {
                0: 0.0002553523698758782,
                1: -0.0009175435313368821,
                2: 0.8213982009306806,
                3: 0.7518493696649671,
            },
            "negative_likelihood_ratio": {
                0: 0.9993730013172721,
                1: 1.0019755695752002,
                2: 0.17393788416266381,
                3: 0.17580241001836563,
            },
            "negative_predictive_value": {
                0: 0.7905518997319063,
                1: 0.6859298775500364,
                2: 0.9559207207300318,
                3: 0.9256372536586603,
            },
            "overall_bias": {
                0: 2.384759178883307,
                1: 1.5915532458962303,
                2: 0.9391095078103142,
                3: 1.0201517776128022,
            },
            "positive_likelihood_ratio": {
                0: 1.0006278388080183,
                1: 0.9980204199585793,
                2: 29.008400096659685,
                3: 10.11946078073544,
            },
            "positive_predictive_value": {
                0: 0.20965595144407367,
                1: 0.3132186886695967,
                2: 0.8849289209482831,
                3: 0.8222004802988526,
            },
            "prevalence": {
                0: 0.2095519699407438,
                1: 0.3136451012841951,
                2: 0.2095519699407438,
                3: 0.3136451012841951,
            },
            "prevalence_threshold": {
                0: 0.49992154477568274,
                1: 0.5002476927296949,
                2: 0.15659390271929277,
                3: 0.23917093832281752,
            },
            "true_negative_rate": {
                0: 0.500334754618301,
                1: 0.5005069929850264,
                2: 0.9713515684894821,
                3: 0.9171132435111808,
            },
            "true_positive_rate": {
                0: 0.4999789546137676,
                1: 0.4985042206274575,
                2: 0.8310451633988546,
                3: 0.8387692815309742,
            },
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "1", 1: "2"},
            "band": {0: "averaged", 1: "averaged"},
            "fn": {0: 270192.0, 1: 241027.0},
            "fp": {0: 804888.0, 1: 463498.0},
            "tn": {0: 2242120.0, 1: 1128240.0},
            "tp": {0: 537586.0, 1: 486353.0},
            "accuracy": {0: 0.7211051404669416, 1: 0.6962099384334907},
            "balanced_accuracy": {0: 0.7006776102801013, 1: 0.6887234346636597},
            "critical_success_index": {0: 0.33335234946355913, 1: 0.408398677278445},
            "equitable_threat_score": {0: 0.1924875708147104, 1: 0.21102455733498296},
            "f_score": {0: 0.5000213928413972, 1: 0.5799475444944674},
            "false_discovery_rate": {0: 0.5995557455861342, 1: 0.4879691656901977},
            "false_negative_rate": {0: 0.33448794099368884, 1: 0.33136324892078417},
            "false_omission_rate": {0: 0.10754715178687997, 1: 0.17602629728168429},
            "false_positive_rate": {0: 0.26415683844610843, 1: 0.29118988175189636},
            "fowlkes_mallows_index": {0: 0.5162368451323665, 1: 0.5851176236495415},
            "matthews_correlation_coefficient": {
                0: 0.34286408564662785,
                1: 0.35612337831296403,
            },
            "negative_likelihood_ratio": {0: 0.4545641768109191, 1: 0.4674922668143933},
            "negative_predictive_value": {0: 0.89245284821312, 1: 0.8239737027183157},
            "overall_bias": {0: 1.6619343433468106, 1: 1.305852511754516},
            "positive_likelihood_ratio": {0: 2.519382284104996, 1: 2.2962224753706137},
            "positive_predictive_value": {0: 0.4004442544138657, 1: 0.5120308343098022},
            "prevalence": {0: 0.2095519699407438, 1: 0.3136451012841951},
            "prevalence_threshold": {0: 0.38650984590178383, 1: 0.39756228103220215},
            "true_negative_rate": {0: 0.7358431615538915, 1: 0.7088101182481037},
            "true_positive_rate": {0: 0.6655120590063112, 1: 0.6686367510792158},
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "averaged", 1: "averaged"},
            "band": {0: "1", 1: "2"},
            "fn": {0: 384342.0, 1: 126877.0},
            "fp": {0: 1158773.0, 1: 109613.0},
            "tn": {0: 1160600.0, 1: 2209760.0},
            "tp": {0: 383237.0, 1: 640702.0},
            "accuracy": {0: 0.5001169438332699, 1: 0.9233904511634777},
            "balanced_accuracy": {0: 0.4998370304129852, 1: 0.8937226034249874},
            "critical_success_index": {0: 0.19894442967848036, 1: 0.7304010980492298},
            "equitable_threat_score": {
                0: -0.00012183020636716884,
                1: 0.6575706128971429,
            },
            "f_score": {0: 0.3318659726903791, 1: 0.8441986067538313},
            "false_discovery_rate": {0: 0.7514691863217489, 1: 0.1460893091568208},
            "false_negative_rate": {0: 0.5007197956171286, 1: 0.16529503803517293},
            "false_omission_rate": {0: 0.2487743876469149, 1: 0.05429897754764647},
            "false_positive_rate": {0: 0.4996061435569009, 1: 0.04725975511485216},
            "fowlkes_mallows_index": {0: 0.3522591594958441, 1: 0.8442532148127214},
            "matthews_correlation_coefficient": {
                0: -0.00028176284027775196,
                1: 0.7935051423750595,
            },
            "negative_likelihood_ratio": {
                0: 1.0006513652592508,
                1: 0.17349433796102434,
            },
            "negative_predictive_value": {0: 0.751225612353085, 1: 0.9457010224523535},
            "overall_bias": {0: 2.0089267684498924, 1: 0.9775085040106621},
            "positive_likelihood_ratio": {0: 0.999347607754162, 1: 17.662067015292408},
            "positive_predictive_value": {0: 0.2485308136782511, 1: 0.8539106908431792},
            "prevalence": {0: 0.24865271633637323, 1: 0.24865271633637323},
            "prevalence_threshold": {0: 0.5000815756425587, 1: 0.19221061838397122},
            "true_negative_rate": {0: 0.5003938564430991, 1: 0.9527402448851479},
            "true_positive_rate": {0: 0.49928020438287135, 1: 0.834704961964827},
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "1", 1: "1", 2: "2", 3: "2"},
            "band": {0: "1", 1: "2", 2: "1", 3: "2"},
            "mean_percentage_error": {
                0: 0.12592779099941254,
                1: -0.11184360086917877,
                2: 0.16711626946926117,
                3: -0.14318732917308807,
            },
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "1", 1: "2"},
            "band": {0: "averaged", 1: "averaged"},
            "mean_percentage_error": {0: 0.007042095065116882, 1: 0.011964470148086548},
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "averaged", 1: "averaged"},
            "band": {0: "1", 1: "2"},
            "mean_percentage_error": {0: 0.14652203023433685, 1: -0.12751546502113342},
        }
    ),
    pd.DataFrame(
        {
            "subsample": {0: "averaged", 1: "averaged"},
            "band": {0: "1", 1: "2"},
            "mean_percentage_error": {0: 0.08395186066627502, 1: -0.03728120028972626},
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
