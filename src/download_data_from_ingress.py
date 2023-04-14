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

# %%
# Load packages
from google.cloud import storage  # interact with data buckets

# Initialise client and note bucket location
client = storage.Client()
bucket = client.bucket("ons-net-zero-data-prod-net-zero-somalia-des-ingress")
bucket_prefix = "ons-des-prod-net-zero-somalia-ingress/"

# %%
# Get all blobs in ingress area
# - Blob is a Binary Large OBject (BLOB) is a collection of binary data stored as a single entity
blobs = list(bucket.list_blobs(prefix=bucket_prefix))

# Note folder of interest in ingress area and filter blobs
ingress_folder_of_interest = "training_data/" # slash important here as training_data_doolow folder stil in ingress
blobs = [blob for blob in blobs if ingress_folder_of_interest in blob.name]

# Print all blobs available
out = [print(blob.name) for blob in blobs]

# %%
# Note local data folder path
data_folder = Path.cwd().parent.joinpath("data")

# Examine each blob
for blob in blobs:
    
    # Get file name
    file_path = Path(blob.name)
    file_path = file_path.relative_to(*file_path.parts[:1]) # Trim bucket prefix
    file_path = data_folder.joinpath(file_path)
    
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
list(data_folder.iterdir())

# %%
