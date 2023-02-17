import dask
import dask.dataframe as dd
import xarray

@dask.delayed()
def crosstab_rasters(
    candidate_map: xarray.DataArray,
    benchmark_map: xarray.DataArray
    ) -> tuple[dask.dataframe.DataFrame,dask.array.Array]:
    
    """
    Creates contingency and agreement tables as dds from candidate and benchmark sliceable arrays.

    Only to be used on spatially aligned candidates and benchmarks.
    """

    """
    Another alternative:
        Checkout groupby operations for xarray:
            - https://docs.xarray.dev/en/stable/user-guide/groupby.html#split
            - Consider doing inner join on x,y coords for already aligned candidate and benchmarks.
            - Then do groupby operation on candidate and benchmark variables \ 
                    with groupby_obj = merged.groupby(['candidate','benchmark'])
            - Consider sorting groupby_object in operation above
            - Checkout methods associated with groupby objects in xarray:
                - https://docs.xarray.dev/en/stable/api.html?highlight=groupby#groupby-objects
            - Access groups with groupby_obj.groups or list(groupby_obj)
            - Try getting length of groupby_obj with nunique = len(groupby_obj) 
            - With length of groupby_obj, create a new array that will be used as a new variable as uniq_group_idx = np.arange(nunique)
            - Map the array using cross_tab_xr = groupby_obj.map(lambda x : unique_group_idx.pop(0))
            - This creates a cross tabulation xarray Dataset
        - Crosstab table
            - Convert crosstab xr ds to dask df.
            - Consider using dask dataframe pivot_table with crosstab xarray Dataset
            - This should yield crosstab df table
    """


    # convert to dask dataframes with only the data via a dataset
    # only use indices from benchmark
    candidate_map_dd = candidate_map.to_dataset(name='candidate').to_dask_dataframe().loc[:,'candidate']
    benchmark_map_dd = benchmark_map.to_dataset(name='benchmark').to_dask_dataframe().loc[:,:]

    # concat dds
    comparison_dd = dd.concat([candidate_map_dd, benchmark_map_dd],axis=1)

    # create categorical datatypes
    comparison_dd = comparison_dd.categorize(columns=['benchmark','candidate'])

    # nans index
    comparison_dd_no_nans = comparison_dd.dropna()
    #breakpoint()

    # create contingency table with ascending categories
    contingency_table = comparison_dd.value_counts(ascending=True)

    # create agreement table, extract count column, and convert to dask Array
    agreement_table = contingency_table.reset_values(name='count').loc[:,'count'].to_dask_array()

    # convert agreement table back to dask array
    agreement_array = None

    """
    Alternative idea based on pandas ngroup:
    https://stackoverflow.com/questions/71702062/for-dask-is-there-something-equivalent-to-ngroup-that-is-not-coumcount

    import pandas as pd
    import dask.dataframe as dd

    df = pd.DataFrame({
        'x': list('aabbcd'),
    })
    ddf = dd.from_pandas(df, npartitions=2)

    nuniq = ddf['x'].nunique().compute()
    c = list(range(nuniq+1))

    ddf.groupby("x").apply(lambda g: g.assign(y = lambda x: c.pop(0)), meta={'x': 'f8', 'y': 'f8'}).compute()
    """

    return(agreement_table, contingency_table)


def compare_continuous_rasters():
    pass
