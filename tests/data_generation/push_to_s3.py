import boto3
from glob import glob


def upload_to_s3(lzw_dir: str, bucket: str):
    """
    Pushes Cloud Optimized GeoTiffs to S3

    Parameters
    ----------
    lzw_dir: str
        Directory of lzws compressed COGs
    bucket: str
        S3 bucket to place files in
    """

    files = glob(f"{lzw_dir}/*.tif")
    s3 = boto3.client("s3")

    for fil in files:
        fil_split = fil.split("/")[-1]

        with open(fil, "rb") as f:
            object_data = f.read()
            s3.put_object(Body=object_data, Bucket=bucket, Key=fil_split)


if __name__ == "__main__":
    upload_to_s3("../../data/data/lzw", "gval-test")
