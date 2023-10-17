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

# %% [markdown]
# # Download data from ingress
#
# #### Purpose
# To download data from the ingress bucket to local GCP storage.
#
# #### Things to note
# * We do not have necessary permissions to overwrite or delete files in the ingress bucket, therefore, this ingress process uses the logic of downloading folders with the latest date (`training/validation_data_YY/MM/DD`), deleting all old files in the local folders stoarage (including `.npy`), and downloading data into either `training` or `validation` local folders (with no date prefix).
# * The kernel should be `Python 3` unlike other notebooks in this workflow.
# * This notebook only needs run when new training or validation data has been ingressed. If this notebook has been run then `premodelling_notebook.py` must be run next to create the `.npy` files.
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
import ipywidgets as widgets
from IPython.display import display

# load functions library
from functions_library import get_folder_paths

# %%
# for reading in config.yaml directories
folder_dict = get_folder_paths()

# Note local data folder path - We always want it to end up in training_data or validation_data
# rather than the specific timestamp.
training_data_dir = Path(folder_dict["training_dir"])
validation_data_dir = Path(folder_dict["validation_dir"])

# %%
# Initialise client and note bucket location
client = storage.Client()
bucket = client.bucket("ons-net-zero-data-prod-net-zero-somalia-des-ingress")
bucket_prefix = "ons-des-prod-net-zero-somalia-ingress/"

# %% [markdown]
# ### Select whether you want to download the latest training/validation

# %%
folders = ["validation_data", "training_data"]
folder_dropdown = widgets.Dropdown(options=folders, description="select folder:")
display(folder_dropdown)

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
        # Get base ingest folder name - ie. training_data_20230927
        folder_name = blob.name.split("/")
        folder_name = folder_name[1]

        # Check if Folder name matches selected drop_down widget
        if folder_name.startswith(folder_dropdown.value):
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
# ### Remove existing files from local folders <a name="remove"></a>

# %%
if folder_dropdown.value == "training_data":
    data_dir = training_data_dir
elif folder_dropdown.value == "validation_data":
    data_dir = validation_data_dir


# %%
def rm_tree(pth):
    pth = Path(pth)
    if pth.exists():
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
