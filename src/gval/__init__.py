from gval.comparison.compute_comparison import ComparisonProcessing
from gval.statistics.categorical_statistics import CategoricalStatistics
from gval.statistics.continuous_statistics import ContinuousStatistics

Comparison = ComparisonProcessing()
CatStats = CategoricalStatistics()
ContStats = ContinuousStatistics()

from gval.accessors import gval_array, gval_dataset, gval_dataframe
from gval.catalogs.catalogs import catalog_compare
