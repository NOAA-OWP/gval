import os
import boto3
from config import TEST_DATA

ESSENTIAL_FILES = [
    "benchmark_map_0.tif",
    "candidate_map_0.tif",
    "agreement_map_szudzik.tif",
    "agreement_map_cantor.tif",
]

for file in ESSENTIAL_FILES:
    if not os.path.exists(os.path.join(TEST_DATA, file)):
        print("Downloading: ", file)
        s3 = boto3.client("s3")
        s3.download_file("gval-test", file, os.path.join(TEST_DATA, file))
