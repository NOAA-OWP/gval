from pytest_cases import parametrize

from tests.conftest import _load_xarray


plot_maps = [
    _load_xarray("categorical_multiband_4.tif", mask_and_scale=True),
    _load_xarray("categorical_multiband_6.tif", mask_and_scale=True),
    _load_xarray(
        "categorical_multiband_8.tif", mask_and_scale=True, band_as_variable=True
    ).drop_vars("band_8"),
    _load_xarray("categorical_multiband_10.tif", mask_and_scale=True),
    _load_xarray("categorical_multiband_4.tif", mask_and_scale=True).sel(
        {"band": [1, 2, 3]}
    ),
    _load_xarray("categorical_multiband_4.tif", mask_and_scale=True).sel(
        {"band": 1, "x": -169443.7041}, method="nearest"
    ),
]
candidate_maps = [
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True),
    _load_xarray("candidate_map_0_accessor.tif", mask_and_scale=True).sel(
        band=1, drop=True
    ),
]


@parametrize(
    "candidate_map, crs, entries",
    list(
        zip(
            [
                candidate_maps[0],
                candidate_maps[1],
                plot_maps[0],
                plot_maps[1],
                plot_maps[2],
            ],
            [
                "EPSG:5070",
                """PROJCS["NAD27 / California zone II",
                                       GEOGCS["GCS_North_American_1927",DATUM["D_North_American_1927",
                                       SPHEROID["Clarke_1866",6378206.4,294.9786982138982]],
                                       PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],
                                       PROJECTION["Lambert_Conformal_Conic_2SP"],
                                       PARAMETER["standard_parallel_1",39.83333333333334],
                                       PARAMETER["standard_parallel_2",38.33333333333334],
                                       PARAMETER["latitude_of_origin",37.66666666666666],
                                       PARAMETER["central_meridian",-122],
                                       PARAMETER["false_easting",2000000],
                                       PARAMETER["false_northing",0],UNIT["Foot_US",0.30480060960121924]]""",
                "EPSG:5070",
                "EPSG:5070",
                "EPSG:5070",
            ],
            [2, 2, 3, 3, 4],
        )
    ),
)
def case_categorical_plot_success(candidate_map, crs, entries):
    return candidate_map, crs, entries


@parametrize(
    "candidate_map, legend_labels, num_classes",
    list(
        zip(
            [candidate_maps[0], candidate_maps[0], plot_maps[3]],
            [None, ["a", "b", "c"], ["a", "b"]],
            [30, 2, 2],
        )
    ),
)
def case_categorical_plot_fail(candidate_map, legend_labels, num_classes):
    return candidate_map, legend_labels, num_classes


@parametrize(
    "candidate_map, axes",
    list(zip([candidate_maps[0], plot_maps[4]], [2, 6])),
)
def case_continuous_plot_success(candidate_map, axes):
    return candidate_map, axes


@parametrize(
    "candidate_map",
    [plot_maps[3], plot_maps[5]],
)
def case_continuous_plot_fail(candidate_map):
    return candidate_map
