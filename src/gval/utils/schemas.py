"""
DataFrame Schemas with Pandera.
"""

# __all__ = ['*']
__author__ = "Fernando Aristizabal"

from typing import List, Optional

import pandas as pd
import pandera as pa
from pandera.typing import Series, Index, Int64

"""
# TODO:
- Define custom datatypes that are unions of base types?
    - several of the float types below could be set to numeric
    - the band column could be set to numeric or str to allow for datasets.
"""


class Xrspatial_crosstab_df(pa.DataFrameModel):  # pragma: no cover
    """Defines the schema for output of `xrspatial.zonal.crosstab()`"""

    zone: Series[float]
    # TODO: Set schema to include any number of columns.
    # columns: pa.Column(name=Any, dtype=float, nullable=True, checks=None)
    # columns: Series[float] = pa.Field(regex=r"^(0|[1-9][0-9]*)$")

    class Config:
        coerce = True
        strict = False


class Sample_identifiers(pa.DataFrameModel):  # pragma: no cover
    """Crosstab DF schema"""

    band: Series[str]
    # this is where we extend these identifiers to map or catalog level identifiers.
    # NOTE: Sample identifiers are currently columns. Should they be indices instead???
    idx: Index[Int64]

    class Config:
        coerce = True
        strict = True

    @classmethod
    def columns(cls) -> List[str]:
        """Gives access to columns from DataFrameModel"""
        return list(super().to_schema().columns.keys())


class Crosstab_df(Sample_identifiers):  # pragma: no cover
    """Crosstab DF schema"""

    candidate_values: Series[float]
    benchmark_values: Series[float]
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
    tp: Optional[Series[float]]
    tn: Optional[Series[float]] = pa.Field(nullable=True)
    fp: Optional[Series[float]]
    fn: Optional[Series[float]]

    class Config:
        coerce = True
        ordered = False  # this is TEMP. Should set id's to indices?


class Metrics_df(Conditions_df):  # pragma: no cover
    """Metrics DF schema"""

    class Config:
        coerce = True
        strict = False  # set to False, bc columns could include any number of metrics
