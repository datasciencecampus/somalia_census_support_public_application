# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Load packages
from google.cloud import storage # interact with data buckets
from pathlib import Path  # handling file paths

# Initialise client and note bucket location
client = storage.Client()
bucket = client.bucket('ons-net-zero-data-prod-net-zero-somalia-des-ingress')

# +
# Get all blobs in ingress area
# - Blob is a Binary Large OBject (BLOB) is a collection of binary data stored as a single entity
blobs = list(bucket.list_blobs(prefix='ons-des-prod-net-zero-somalia-ingress/'))

# Print all blobs available
out = [print(blob.name) for blob in blobs]

# +
# Set location to copy blobs into local files
path_to_data_folder = Path("../data/training_data_doolow/")

# Examine each blob
for blob in blobs:
    
    # Get file name
    file_name = Path(blob.name).name
    local_file_path = Path(path_to_data_folder, file_name)
    
    # Copy file to local environment
    blob.download_to_filename(local_file_path)
    
    # Print progress
    print(f'Blob ({blob.name}) copied to local environment at {local_file_path}')
# -

read_csv(output_file_name)
