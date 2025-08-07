"""
DataFrame Schemas with Pandera.
"""

from typing import List, Optional, Union

__author__ = "Fernando Aristizabal"

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series, Index, Int64
from shapely import Geometry


class Xrspatial_crosstab_df(pa.DataFrameModel):  # pragma: no cover
    """Defines the schema for output of `xrspatial.zonal.crosstab()`"""

    zone: Series[float]
    # TODO: Set schema to include any number of columns.
    # columns: pa.Column(name=Any, dtype=float, nullable=True, checks=None)
    # columns: Series[float] = pa.Field(regex=r"^(0|[1-9][0-9]*)$")

    class Config:
        coerce = True
        strict = False


class columns_method(pa.DataFrameModel):  # pragma: no cover
    """Defines base data frame model with columns method"""

    @classmethod
    def columns(cls) -> List[str]:
        """Gives access to columns from DataFrameModel"""
        return list(super().to_schema().columns.keys())


class Sample_identifiers(columns_method):  # pragma: no cover
    """Crosstab DF schema"""

    band: Series[str]
    # this is where we extend these identifiers to map or catalog level identifiers.
    # NOTE: Sample identifiers are currently columns. Should they be indices instead???
    # Add subsample
    idx: Index[Int64]

    class Config:
        coerce = True
        strict = True


class Subsample_identifiers(columns_method):  # pragma: no cover
    """Crosstab DF schema"""

    subsample: Optional[Series[str]]
    # this is where we extend these identifiers to map or catalog level identifiers.
    # NOTE: Sample identifiers are currently columns. Should they be indices instead???
    # Add subsample
    idx: Index[Int64]

    class Config:
        coerce = True
        strict = True


class Crosstab_df(Sample_identifiers, Subsample_identifiers):  # pragma: no cover
    """Crosstab DF schema

    Inherits columns from `Sample_identifiers`
    """

    candidate_values: Series = pa.Field(nullable=True)
    benchmark_values: Series = pa.Field(nullable=True)
    agreement_values: Optional[Series[float]]
    counts: Series[float]

    class Config:
        coerce = True
        strict = True


class Pivoted_crosstab_df(pa.DataFrameModel):  # pragma: no cover
    """Pivoted Crosstab DF schema"""

    row_idx: Index[Int64] = pa.Index(name="candidate_values")
    col_idx: Series[str]

    class Config:
        coerce = True
        strict = True

    @pa.dataframe_check
    def column_index_name(cls, df: pd.DataFrame) -> Series[bool]:
        """Checks that column index name is 'benchmark_values'"""
        return df.columns.name == "benchmark_values"


class Conditions_df(Sample_identifiers):  # pragma: no cover
    """Cateogrical conditions df

    Inherits columns from `Sample_identifiers`
    """

    tp: Optional[Series[float]]
    tn: Optional[Series[float]] = pa.Field(nullable=True)
    fp: Optional[Series[float]]
    fn: Optional[Series[float]]

    class Config:
        coerce = True
        ordered = False  # this is TEMP. Should set id's to indices?


class Metrics_df(Conditions_df):  # pragma: no cover
    """Metrics DF schema

    Inherits columns from `Conditions_df`
    """

    class Config:
        coerce = True
        strict = False  # set to False, bc columns could include any number of metrics


class Prob_metrics_df(Sample_identifiers, Subsample_identifiers):  # pragma: no cover
    """Probabilistic metrics DF schema"""

    metrics: Optional[Series[object]]

    class Config:
        coerce = False
        strict = False  # set to False, bc columns could include any number of metrics


class AttributeTrackingDf(pa.DataFrameModel):  # pragma: no cover
    """
    Defines the schema for output of `_attribute_tracking_xarray()`
    The attributes could be of any datatype. For instance, if your attributes are of float type,
    you can use Series[float] instead of Series[object].
    """

    # This is a basic schema considering attributes as object datatype.
    # If your attributes are of specific datatypes, you can modify accordingly.
    # Replace 'attribute_1', 'attribute_2' with your attribute names

    attribute_1_candidate: Optional[Series[object]]
    attribute_2_candidate: Optional[Series[object]]
    attribute_1_benchmark: Optional[Series[object]]
    attribute_2_benchmark: Optional[Series[object]]

    class Config:
        coerce = True
        strict = False

    @pa.dataframe_check
    def validate_column_suffixes(
        cls, df: pd.DataFrame, candidate_suffix: str, benchmark_suffix: str
    ) -> Series[bool]:
        """
        Checks that each column name in the dataframe ends with either '_candidate' or '_benchmark'.
        """
        return all(
            name.endswith((candidate_suffix, benchmark_suffix)) for name in df.columns
        )


class SubsamplingDf(pa.DataFrameModel):  # pragma: no cover
    """
    Defines the schema for subsampling DataFrame`
    """

    subsample_id: Series[int]
    geometry: Series[Geometry]
    subsample_type: Series[str]
    weights: Optional[Union[float, int]]  # Possibly for the future for sample weighting

    class Config:
        coerce = True
        strict = False
