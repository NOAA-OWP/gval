import subprocess
from glob import glob


def lzw_compress(generate_dir: str, lzw_dir: str):
    """
    Creates lzw compressed Cloud Optimized GeoTiffs and places them in a data folder

    Parameters
    ----------
    generate_dir: str
        Directory where data was generated
    lzw_dir: str
        Directory to store lzw compressed COGs

    """
    files = glob(f"{generate_dir}/*.tif")

    for fil in files:
        lzw_fil = f'{lzw_dir}/{fil.split("/")[-1]}'
        subprocess.call(
            f"rio cogeo create {fil} {lzw_fil} --cog-profile lzw", shell=True
        )


if __name__ == "__main__":
    lzw_compress("../../data/data/generate", "../../data/data/lzw")
