import numpy as np
from pytest_cases import parametrize_with_cases
from pytest import raises

# added comment because of flake8 believes it's unused
import gval  # noqa: F401


@parametrize_with_cases(
    "candidate_map, crs, entries",
    glob="categorical_plot_success",
)
def test_categorical_plot_success(candidate_map, crs, entries):
    candidate_map.rio.write_crs(crs, inplace=True)
    viz_object = candidate_map.gval.cat_plot(basemap=None)
    assert len(viz_object.axes.get_legend().texts) == entries


@parametrize_with_cases(
    "candidate_map, legend_labels, num_classes",
    glob="categorical_plot_fail",
)
def test_categorical_plot_fail(candidate_map, legend_labels, num_classes):
    candidate_map.data = np.random.choice(np.arange(num_classes), candidate_map.shape)
    with raises(ValueError):
        _ = candidate_map.gval.cat_plot(legend_labels=legend_labels, basemap=None)


@parametrize_with_cases(
    "candidate_map, axes",
    glob="continuous_plot_success",
)
def test_continuous_plot_success(candidate_map, axes):
    viz_object = candidate_map.gval.cont_plot(basemap=None)
    assert len(viz_object.figure.axes) == axes


@parametrize_with_cases(
    "candidate_map",
    glob="continuous_plot_fail",
)
def test_continuous_plot_fail(candidate_map):
    with raises(ValueError):
        _ = candidate_map.gval.cont_plot()
