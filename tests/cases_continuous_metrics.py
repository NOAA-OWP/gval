"""
Cases for testing continuous value metrics.
"""

# __all__ = ['*']
__author__ = ["Fernando Aristizabal"]

from pytest_cases import parametrize

from tests.conftest import _load_xarray


candidate_maps = ["candidate_continuous_0.tif", "candidate_continuous_1.tif"]
benchmark_maps = ["benchmark_continuous_0.tif", "benchmark_continuous_1.tif"]

all_load_options = [
    {"mask_and_scale": True},
    {"mask_and_scale": True, "band_as_variable": True},
]

ev_mean_absolute_error = [
    0.3173885941505432,
    {"band_1": 0.48503121733665466, "band_2": 0.48503121733665466},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(zip(candidate_maps, benchmark_maps, all_load_options, ev_mean_absolute_error)),
)
def case_mean_absolute_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_mean_squared_error = [
    0.30277130007743835,
    {"band_1": 0.5341722369194031, "band_2": 0.5341722369194031},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(zip(candidate_maps, benchmark_maps, all_load_options, ev_mean_squared_error)),
)
def case_mean_squared_error(candidate_map, benchmark_map, load_options, expected_value):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_root_mean_squared_error = [
    0.5502465963363647,
    {"band_1": 0.7308709025382996, "band_2": 0.7308709025382996},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(
            candidate_maps, benchmark_maps, all_load_options, ev_root_mean_squared_error
        )
    ),
)
def case_root_mean_squared_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_mean_signed_error = [
    0.02950434572994709,
    {"band_1": -0.3584839105606079, "band_2": 0.3584839105606079},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(zip(candidate_maps, benchmark_maps, all_load_options, ev_mean_signed_error)),
)
def case_mean_signed_error(candidate_map, benchmark_map, load_options, expected_value):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_mean_percentage_error = [
    0.015025136061012745,
    {"band_1": -0.1585608273744583, "band_2": 0.1368602067232132},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(candidate_maps, benchmark_maps, all_load_options, ev_mean_percentage_error)
    ),
)
def case_mean_percentage_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_mean_absolute_percentage_error = [
    0.15956786274909973,
    {"band_1": 0.20223499834537506, "band_2": 0.15323485434055328},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            all_load_options,
            ev_mean_absolute_percentage_error,
        )
    ),
)
def case_mean_absolute_percentage_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_mean_normalized_root_absolute_error = [
    0.2802138924598694,
    {"band_1": 0.32327112555503845, "band_2": 0.2790282368659973},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            all_load_options,
            ev_mean_normalized_root_absolute_error,
        )
    ),
)
def case_mean_normalized_root_mean_squared_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_range_normalized_root_mean_squared_error = [
    0.4702962636947632,
    {"band_1": 0.8913060426712036, "band_2": 0.42992404103279114},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            all_load_options,
            ev_range_normalized_root_mean_squared_error,
        )
    ),
)
def case_range_normalized_root_mean_squared_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_mean_normalized_mean_absolute_error = [
    0.16163060069084167,
    {"band_1": 0.21453391015529633, "band_2": 0.18517279624938965},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            all_load_options,
            ev_mean_normalized_mean_absolute_error,
        )
    ),
)
def case_mean_normalized_mean_absolute_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_range_normalized_mean_absolute_error = [
    0.27127230167388916,
    {"band_1": 0.5915015339851379, "band_2": 0.2853124737739563},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            all_load_options,
            ev_range_normalized_mean_absolute_error,
        )
    ),
)
def case_range_normalized_mean_absolute_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_coefficient_of_determination = [
    -0.06615996360778809,
    {"band_1": -2.829420804977417, "band_2": 0.10903036594390869},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            all_load_options,
            ev_coefficient_of_determination,
        )
    ),
)
def case_coefficient_of_determination(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )


ev_symmetric_mean_absolute_percentage_error = [
    0.1628540771054295,
    {"band_1": 0.19877496825251173, "band_2": 0.19877496825251173},
]


@parametrize(
    "candidate_map, benchmark_map, load_options, expected_value",
    list(
        zip(
            candidate_maps,
            benchmark_maps,
            all_load_options,
            ev_symmetric_mean_absolute_percentage_error,
        )
    ),
)
def case_symmetric_mean_absolute_percentage_error(
    candidate_map, benchmark_map, load_options, expected_value
):
    return (
        _load_xarray(candidate_map, **load_options),
        _load_xarray(benchmark_map, **load_options),
        expected_value,
    )
