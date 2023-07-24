"""
Carries over attributes from xarray's DataArray and Dataset classes.

TODO:
    - [ ] implement tests with xr.testing.assert_identical
    - [ ] make additional test cases???
    - [ ] add to accessor methods
    - [ ] documentation
"""

__author__ = "Fernando Aristizabal"

from typing import Optional, Iterable, Union, Tuple

import pandas as pd
import xarray as xr
from pandera.typing import DataFrame

from gval.utils.schemas import AttributeTrackingDf


def _attribute_tracking_xarray(
    candidate_map: Union[xr.DataArray, xr.Dataset],
    benchmark_map: Union[xr.DataArray, xr.Dataset],
    agreement_map: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    candidate_suffix: Optional[str] = "_candidate",
    benchmark_suffix: Optional[str] = "_benchmark",
    candidate_include: Optional[Iterable[str]] = None,
    candidate_exclude: Optional[Iterable[str]] = None,
    benchmark_include: Optional[Iterable[str]] = None,
    benchmark_exclude: Optional[Iterable[str]] = None,
) -> Union[
    DataFrame[AttributeTrackingDf],
    Tuple[DataFrame[AttributeTrackingDf], Union[xr.DataArray, xr.Dataset]],
]:
    """
    Concatenate xarray attributes into a single pandas dataframe.

    Parameters
    ----------
    candidate_map : Union[xr.DataArray, xr.Dataset]
        Candidate map xarray object.
    benchmark_map : Union[xr.DataArray, xr.Dataset]
        Benchmark map xarray object.
    candidate_suffix : Optional[str], default = '_candidate'
        Suffix to append to candidate map xarray attributes, by default '_candidate'.
    benchmark_suffix : Optional[str], default = '_benchmark'
        Suffix to append to benchmark map xarray attributes, by default '_benchmark'.
    candidate_include : Optional[Iterable[str]], default = None
        List of attributes to include from candidate map. candidate_include and candidate_exclude are mutually exclusive arguments.
    candidate_exclude : Optional[Iterable[str]], default = None
        List of attributes to exclude from candidate map. candidate_include and candidate_exclude are mutually exclusive arguments.
    benchmark_include : Optional[Iterable[str]], default = None
        List of attributes to include from benchmark map. benchmark_include and benchmark_exclude are mutually exclusive arguments.
    benchmark_exclude : Optional[Iterable[str]], default = None
        List of attributes to exclude from benchmark map. benchmark_include and benchmark_exclude are mutually exclusive arguments.

    Raises
    ------
    ValueError
        If candidate_include and candidate_exclude are both not None.
    ValueError
        If benchmark_include and benchmark_exclude are both not None.

    Returns
    -------
    Union[DataFrame[AttributeTrackingDf], Tuple[DataFrame[AttributeTrackingDf], Union[xr.DataArray, xr.Dataset]]]
        Pandas dataframe with concatenated attributes from candidate and benchmark maps. If agreement_map is not None, returns a tuple with the dataframe and the agreement map.
    """

    candidate_attrs = candidate_map.attrs
    benchmark_attrs = benchmark_map.attrs

    # remove default exclude from both includes
    if (candidate_include is not None) and (candidate_exclude is not None):
        raise ValueError(
            "candidate_include and candidate_exclude are mutually exclusive"
        )

    if (benchmark_include is not None) and (benchmark_exclude is not None):
        raise ValueError(
            "benchmark_include and benchmark_exclude are mutually exclusive"
        )

    # candidate include and exclude
    if candidate_include is not None:
        candidate_attrs = {
            k: v for k, v in candidate_attrs.items() if k in candidate_include
        }
    elif candidate_exclude is not None:
        candidate_attrs = {
            k: v for k, v in candidate_attrs.items() if k not in candidate_exclude
        }

    # benchmark include and exclude
    if benchmark_include is not None:
        benchmark_attrs = {
            k: v for k, v in benchmark_attrs.items() if k in benchmark_include
        }
    elif benchmark_exclude is not None:
        benchmark_attrs = {
            k: v for k, v in benchmark_attrs.items() if k not in benchmark_exclude
        }

    # Convert xarray attributes to pandas dataframes
    candidate_df = pd.DataFrame([candidate_attrs], index=[0])
    benchmark_df = pd.DataFrame([benchmark_attrs], index=[0])

    # Append a suffix to each dataframe's column names to denote its origin
    candidate_df.columns = [f"{col}{candidate_suffix}" for col in candidate_df.columns]
    benchmark_df.columns = [f"{col}{benchmark_suffix}" for col in benchmark_df.columns]

    # Concatenate the dataframes together
    combined_df = pd.concat([candidate_df, benchmark_df], axis=1)

    # validate schema
    AttributeTrackingDf.validate_column_suffixes(
        combined_df, candidate_suffix, benchmark_suffix
    )

    if agreement_map is None:
        return combined_df

    updated_candidate_attrs = candidate_df.to_dict(orient="records")[0]
    updated_benchmark_attrs = benchmark_df.to_dict(orient="records")[0]

    agreement_map = agreement_map.assign_attrs(
        {**updated_candidate_attrs, **updated_benchmark_attrs}
    )

    return combined_df, agreement_map
