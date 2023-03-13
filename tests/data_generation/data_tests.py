from itertools import product

import numpy as np
from tqdm import tqdm
import xarray as xr

from gval.homogenize.spatial_alignment import Spatial_alignment
from gval.utils.loading_datasets import load_raster_as_xarray
from gval.utils.exceptions import RasterMisalignment
from gval.compare import (
    compute_agreement_xarray,
    cantor_pair_signed,
    szudzik_pair_signed,
)
import shutil


def generate_aligned_and_agreement_maps(
    candidate_map: str,
    benchmark_map: str,
    target_map: str,
    data_dir: str,
    generate_dir: str,
):
    """
    Generates test data from base candidate and benchmark maps

    Parameters
    ----------
    candidate_map: str
        Path for the candidate map
    benchmark_map: str
        Path for the benchmark map
    target_map: str
        Path for the target map
    data_dir: str
        Directory to read data from
    generate_dir: str
        Directory to generate data

    """

    target_maps = ["candidate", "benchmark", "target"]

    comparison_functions = [
        (cantor_pair_signed, "cantor"),
        (szudzik_pair_signed, "szudzik"),
        ("pairing_dict", "pairing"),
    ]

    # Load rasters
    candidate, benchmark = load_raster_as_xarray(
        f"{data_dir}/{candidate_map}"
    ), load_raster_as_xarray(f"{data_dir}/{benchmark_map}")

    names = [candidate_map, benchmark_map, target_map]

    # For use in identifying where agreement rasters were based
    agreement_numbers = candidate_map[-5] + benchmark_map[-5]

    # For every target map and
    for idx, map_al in enumerate(tqdm(target_maps)):
        try:
            align_arg = (
                load_raster_as_xarray(f"{data_dir}/{target_map}")
                if map_al == "target"
                else map_al
            )

            cam, bem = Spatial_alignment(candidate, benchmark, align_arg)
            align_string = f"aligned_to_{names[idx][:-4]}"

            cam.rio.to_raster(f"{generate_dir}/{candidate_map[:-4]}_{align_string}.tif")
            bem.rio.to_raster(f"{generate_dir}/{benchmark_map[:-4]}_{align_string}.tif")

            for comp in comparison_functions:
                allow_candidate_values = [-9999, 1, 2] if comp[1] == "pairing" else None
                allow_benchmark_values = [0, 2] if comp[1] == "pairing" else None

                agreement_map_computed = compute_agreement_xarray(
                    cam,
                    bem,
                    comp[0],
                    allow_candidate_values=allow_candidate_values,
                    allow_benchmark_values=allow_benchmark_values,
                )

                agreement_map_computed.rio.set_crs(cam.rio.crs)

                agreement_map_computed.rio.set_nodata(-9999)
                if np.nan in agreement_map_computed:
                    agreement_map_computed = xr.where(
                        np.isnan(agreement_map_computed), -9999, agreement_map_computed
                    )

                agreement_map_computed.rio.to_raster(
                    f"{generate_dir}/agreement_map_{agreement_numbers}_{comp[1]}_"
                    f"{align_string}.tif"
                )

        except KeyError:
            continue
        except RasterMisalignment:
            continue


if __name__ == "__main__":
    a = np.array(["candidate_map_0.tif", "candidate_map_1.tif"])
    b = np.array(["benchmark_map_0.tif", "benchmark_map_1.tif"])
    c = np.array(["target_map_0.tif", "target_map_1.tif"])

    for cb in product(a, b, c):
        generate_aligned_and_agreement_maps(
            *cb, "../../data/data", "../../data/data/generate"
        )

    for t_map in a:
        shutil.copy(f"../../data/data/{t_map}", f"../../data/data/generate/{t_map}")
    for t_map in b:
        shutil.copy(f"../../data/data/{t_map}", f"../../data/data/generate/{t_map}")
    for t_map in c:
        shutil.copy(f"../../data/data/{t_map}", f"../../data/data/generate/{t_map}")
