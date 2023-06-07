# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#
#
# ### Contents
#
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Download from last modified date](#moddate)
# 1. ##### [Remove existing files](#remove)
# 1. ##### [Download data](#download)
# 1. ##### [Check files](#checkfile)
# %% [markdown]
# ### Set-up <a name="setup"></a>

# %%
# Load packages
from google.cloud import storage  # interact with data buckets
from pathlib import Path  # handling file paths
from pytz import timezone
import datetime

# Initialise client and note bucket location
client = storage.Client()
bucket = client.bucket("ons-net-zero-data-prod-net-zero-somalia-des-ingress")
bucket_prefix = "ons-des-prod-net-zero-somalia-ingress/"

# Note local data folder path - We always want it to end up in training_data,
# rather than the specific timestamp.
data_dir = Path.cwd().parent.joinpath("data/training_data/")

# %% [markdown]
# ### Download from last modified date <a name="moddate"></a>

# %%
# Storage for the most recently ingested file path
lastest_object = ""

# Get earliest datetime available (0001-01-01 00:00:00)
latest_modified_time = datetime.datetime.min
# Convert Naive datetime to Timezone encoded to match GCP
latest_modified_time = timezone("UTC").localize(latest_modified_time)


# Iterate through all blobs in ingress/
for blob in bucket.list_blobs(prefix=bucket_prefix):
    # Check each files modified_time
    modified_time = blob.updated
    if modified_time > latest_modified_time:
        latest_object = blob.name
        latest_modified_time = modified_time

# Split string down, take only 2nd part (ons-des-prod-net-zero-somalia-ingress/<target_folder>/...)
path_components = latest_object.split("/")
latest_folder = path_components[1] + "/"

# %%
# Get all blobs in ingress area
# - Blob is a Binary Large OBject (BLOB) is a collection of binary data stored as a single entity
blobs = list(bucket.list_blobs(prefix=bucket_prefix))

# Note folder of interest in ingress area and filter blobs
ingress_dir_of_interest = latest_folder
blobs = [blob for blob in blobs if ingress_dir_of_interest in blob.name]

# Print all blobs available
out = [print(blob.name) for blob in blobs]


# %% [markdown]
# ### Remove existing files from the training_data folders <a name="remove"></a>
# This will remove all .npy files as well!

# %%
def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


rm_tree(data_dir)

# %% [markdown]
# ### Download data <a name="download"></a>

# %%
# Examine each blob
for blob in blobs:
    # Get file name
    file_path = Path(blob.name)
    # Trim bucket prefixes - Each object becomes "img/filename"
    file_path = file_path.relative_to(*file_path.parts[:2])
    file_path = data_dir.joinpath(file_path)

    # Get parent folders for file and recreate structure
    for folder in reversed(file_path.parents):
        # Create folder
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        # Copy file to local environment
    blob.download_to_filename(file_path)

    # Print progress
    print(f"Blob ({blob.name}) copied to local environment at {file_path}")


# %% [markdown]
# ### Check files <a name="checkfiles"></a>

# %%
# Check files are present
list(data_dir.iterdir())

# %%
