import gval
import rioxarray as rxr
import xarray as xr


if __name__ == "__main__":
    agreement_maps = []
    final_dfs = []
    cross_tables = []
    path = "/home/sven/Desktop/Test TIFFS/"
    test_evals = ["12020001", "12020004", "12020007", "12030107", "12030109"]
    differences = []

    for val in test_evals:
        candidate = rxr.open_rasterio(
            f"{path}/inundation_extent_{val}.tif", mask_and_scale=True
        )
        benchmark = rxr.open_rasterio(
            f"{path}/ble_huc_{val}_extent_500yr.tif", mask_and_scale=True
        )

        candidate.values = xr.where(candidate < 0, 0, candidate)
        candidate.values = xr.where(candidate > 0, 1, candidate)

        agreement_map, crosstab, metric_table = candidate.gval.categorical_compare(
            benchmark_map=benchmark,
            target_map="candidate",
            allow_candidate_values=[0, 1],
            allow_benchmark_values=[0, 1],
            negative_categories=[0],
            positive_categories=[1],
        )
