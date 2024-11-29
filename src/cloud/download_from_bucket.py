from pathlib import Path
import shutil
import subprocess
import sys


def download_from_bucket(bucket_path, local_folder):
    """
    Download files from a Google Cloud Storage bucket to a local folder.

    Args:
        bucket_path (str): Path to the Google Cloud Storage bucket.
        local_folder (Path): Path to the local folder where files will be downloaded.
    """
    # Delete existing files in the local folder
    for file in local_folder.glob("*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)

    # Create local directories if they don't exist
    local_folder.mkdir(parents=True, exist_ok=True)

    # Download files from the bucket
    subprocess.run(["gsutil", "-m", "cp", "-r", f"{bucket_path}*", str(local_folder)])


def replicate_folder_structure(bucket_path, local_parent_folder):
    """
    Replicate the folder structure of a Google Cloud Storage bucket locally.

    Args:
        bucket_path (str): Path to the Google Cloud Storage bucket.
        local_parent_folder (Path): Path to the local parent folder where structure will be replicated.
    """
    # Get the folder structure from the bucket
    process = subprocess.Popen(
        ["gsutil", "ls", "-r", bucket_path], stdout=subprocess.PIPE
    )
    output, _ = process.communicate()
    folder_structure = output.decode().splitlines()

    # Create local directories to replicate the bucket structure
    for folder in folder_structure:
        relative_folder_path = folder.replace(bucket_path, "")
        local_folder_path = local_parent_folder / relative_folder_path
        local_folder_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <bucket_path> <local_parent_folder>")
        sys.exit(1)

    bucket_path = sys.argv[1]
    local_parent_folder = Path(sys.argv[2])

    # Replicate folder structure locally
    replicate_folder_structure(bucket_path, local_parent_folder)

    # Set the local folder path for downloading files
    local_folder = local_parent_folder

    # Download files from the bucket to the local folder
    download_from_bucket(bucket_path, local_folder)
