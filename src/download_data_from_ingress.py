# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %%
from pathlib import Path  # handling file paths
import datetime # For finding the most recent folder
from pytz import timezone

# %%
# Load packages
from google.cloud import storage  # interact with data buckets

# Initialise client and note bucket location
client = storage.Client()
bucket = client.bucket("ons-net-zero-data-prod-net-zero-somalia-des-ingress")
bucket_prefix = "ons-des-prod-net-zero-somalia-ingress/"


# %% [markdown]
# ### Get latest folder using String date (YYYYMMDD)

# %%
# Used to stop training_data/ and training_data_doolow/ from throwing errors
def convert_int(s):
    if isinstance(s,str):
        try:
            return int(s)
        except:
            return 0
    else:
        print (s, type(s))
        return None


# %%
# list_blobs() is a lazy loading iterator. It won't populate unless called
# next() makes the first API call. Will throw StopIteration error if bucket is empty!
# Ellipsis is used to prevent this error
blobs = bucket.list_blobs(
    prefix=bucket_prefix,
    delimiter= '/'
)
next(blobs, ...)

# folders = full folder names
# dates = suffix in YYYYMMDD format
folders = []
dates = []
for folder in blobs.prefixes:
    path_components = folder.split('/')
    folders.append(path_components[1] + '/')
    dates.append(folder.split('_')[-1][:-1])

dates = list(map(convert_int, dates))

latest_date = lambda folders, dates: folders[max(range(len(dates)), key=lambda i: dates[i])]
latest_folder = latest_date(folders, dates)

# %% [markdown]
# ### Alternatively use the last modified date

# %%
# Storage for the most recently ingested file path
lastest_object = ''

# Get earliest datetime available (0001-01-01 00:00:00)
latest_modified_time = datetime.datetime.min
# Convert Naive datetime to Timezone encoded to match GCP
latest_modified_time = timezone('UTC').localize(latest_modified_time)


# Iterate through all blobs in ingress/
for blob in bucket.list_blobs(prefix=bucket_prefix):
    # Check each files modified_time
    modified_time = blob.updated
    if modified_time > latest_modified_time:
        latest_object = blob.name
        latest_modified_time = modified_time
            
# Split string down, take only 2nd part (ons-des-prod-net-zero-somalia-ingress/<target_folder>/...)
path_components = latest_object.split('/')
latest_folder = path_components[1] + '/'

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
# ### Remove existing files from the training_data folders
# This will remove all .npy files as well!

# %%
def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()
    
rm_tree(data_dir)

# %%
# Note local data folder path - We always want it to end up in training_data, 
# rather than the specific timestamp.
data_dir = Path.cwd().parent.joinpath("data/training_data/")

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
        if folder.exists() == False: folder.mkdir(parents=True, exist_ok=True)
        # Copy file to local environment
    blob.download_to_filename(file_path)

    # Print progress
    print(f"Blob ({blob.name}) copied to local environment at {file_path}")


# %%
# Check files are present
list(data_dir.iterdir())

# %%
