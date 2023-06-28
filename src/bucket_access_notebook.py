# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Set-up

# %%
# libraries
from google.cloud import storage
from pathlib import Path

# %%
# set data directory
data_dir = Path.cwd().parent.joinpath("data")

# set model and output directories
models_dir = Path.cwd().parent.joinpath("models")
outputs_dir = Path.cwd().parent.joinpath("outputs")

# set training_data directory within data folder
training_data_dir = data_dir.joinpath("training_data")

# set img and mask directories within training_data directory
img_dir = training_data_dir.joinpath("img")
mask_dir = training_data_dir.joinpath("mask")

# %%
# work-in-progress bucket
bucket_name = "ons-net-zero-analysis-prod-somalia-wip"
destination_folder = "test/"


# %% [markdown]
# ### Functions

# %% [markdown]
# #### Files

# %%
def move_file_to_bucket(source_file_path, bucket_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob_name = source_file_path.name
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(source_file_path))
    print(f"File {source_file_path} uploaded to {bucket_name}/{blob_name}")


# %%
def delete_file_from_bucket(bucket_name, file_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(str(file_name))
    blob.delete()
    print(f"File {file_name} deleted from {bucket_name}")


# %%
def read_files_in_bucket(bucket_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()

    for blob in blobs:
        print(f"File: {blob.name}")


# %% [markdown]
# #### Folders

# %%
def move_folder_to_bucket(source_folder_path, bucket_name, destination_folder_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    source_folder_path = Path(source_folder_path)
    destination_folder_name = Path(destination_folder_name)

    for file_path in source_folder_path.glob("**/*"):
        if file_path.is_file():
            relative_file_path = file_path.relative_to(source_folder_path)
            destination_blob_name = destination_folder_name / relative_file_path
            blob = bucket.blob(str(destination_blob_name))
            blob.upload_from_filename(str(file_path))

        print(f"File {file_path} moved to {bucket_name}/{destination_blob_name}")


# %%
def delete_folder_from_bucket(bucket_name, fodler_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=folder_name)

    for blob in blobs:
        blob.delete()
        print(f"File {blob.name} deleted from {bucket_name}")


# %%
def read_files_in_folder(bucket_name, folder_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=folder_name)

    for blob in blobs:
        if not blob.name.endswith("/"):
            print(f"File: {blob.name}")


# %%
def download_folder_from_bucket(bucket_name, folder_name, destination_folder):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=folder_name)
    
    for blob in blobs:
        if not blob.name.endswith('/'):
            relative_path = Path(blob.name).relative_to(folder_name)
            destination_path = Path(destination_folder) / relative_path
            
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(str(destination_path))
            print(f"File {blob.name} downloaded to {destination_path}")


# %% [markdown]
# ### In action

# %% [markdown]
# ### Files

# %%
# for file testing
runid = "outputs_alt_test"
X_test_filename = f"{runid}_xtest.npy"
X_test_path = outputs_dir.joinpath(X_test_filename)

X_test_path

# %%
move_file_to_bucket(X_test_path, bucket_name)

# %%
delete_file_from_bucket(bucket_name, "test/X_test.npy")

# %%
read_files_in_bucket(bucket_name)

# %% [markdown]
# #### Folders

# %%
destination_folder_name = "mask"
move_folder_to_bucket(mask_dir, bucket_name, destination_folder_name)

# %% jupyter={"outputs_hidden": true}
folder_name = "mask"
delete_folder_from_bucket(bucket_name, folder_name)

# %%
folder_name = "mask"
read_files_in_folder(bucket_name, folder_name)

# %%
folder_name = 'mask'
destination_folder = mask_dir
download_folder_from_bucket(bucket_name, folder_name, destination_folder)

# %%
