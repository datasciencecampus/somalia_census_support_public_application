import subprocess
import sys
from pathlib import Path


def upload_to_bucket(local_folder, bucket_path):
    """
    Upload files from a local folder to a Google Cloud Storage bucket.

    Args:
        local_folder (Path): Path to the local folder containing files to be uploaded.
        bucket_path (str): Path to the Google Cloud Storage bucket.
    """
    # Upload files to the bucket
    subprocess.run(["gsutil", "-m", "cp", "-r", f"{local_folder}/*", bucket_path])


if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <local_folder> <bucket_path>")
        sys.exit(1)

    local_folder = Path(sys.argv[1])
    bucket_path = sys.argv[2]

    # Upload files from the local folder to the bucket
    upload_to_bucket(local_folder, bucket_path)
