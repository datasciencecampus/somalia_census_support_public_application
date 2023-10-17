# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Pre-Ingress Notebook
#
# The purpose of this notebook is to check training and validation image (`.tif`) and mask files (`.geojson`) for any potential errors before being ingressed to GCP.
#
# This notebook should be ran until it returns no errors - files will be updated in GIS software.
#
# Before running this notebook, files should be saved locally with the below structure:
#
# ```
# 📦somalia_unfpa_census_support
#  ┣ 📂data
#  ┃ ┣ 📂training_data
#  ┃ ┃ ┗ 📂img
#  ┃ ┃ ┃ ┣ 📜training_data_<area>_<initial>.tif
#  ┃ ┃ ┗ 📂mask
#  ┃ ┃ ┃ ┣ 📜training_data_<area>_<initial>.geojson
#  ┃ ┣ 📂validation_data
#  ┃ ┃ ┗ 📂img
#  ┃ ┃ ┃ ┣ 📜validation_data_<area>_<initial>.tif
#  ┃ ┃ ┗ 📂mask
#  ┃ ┃ ┃ ┣ 📜validation_data_<area>_<initial>.geojson
#  ┣ 📂src
#  ┃ ┣ 📜preingress_functions.py
#  ┃ ┣ 📜preingress_notebook.py
#  ┣ 📜.gitignore
#  ┣ 📜requirements.txt
#  ┣ 📜config.yaml
#  ┗ 📜README.md
# ```
#
#
# > **NOTE** files should not be saved to the Sharepoint GCP ingress folder until they have been through the below process

# %% [markdown]
# ## Contents
#
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Explore](#explore)
# 1. ##### [General file cleaning](#filecleaning)
# 1. ##### [Mask file cleaning](#maskfilecleaning)

# %% [markdown]
# ## Set-up <a name="setup"></a>

# %% [markdown]
# ### Import libraries & functions

# %%
# Load required libraries
from pathlib import Path  # working with file paths

# Local imports
from functions_library import get_folder_paths

from preingress_functions import (
    change_to_lower_case,
    vice_versa_check_mask_file_for_img_file,
    check_naming_convention_upheld,
    cleaning_of_mask_files,
    check_same_number_of_files_present,
    get_file_paths,
)

# %% [markdown]
# ### Set-up directories

# %%
# get folder paths from config.yaml
folder_dict = get_folder_paths()
# list of folder names
folder_name = [
    "training_img_dir",
    "training_mask_dir",
    "validation_img_dir",
    "validation_mask_dir",
]
# set folder paths
training_img_dir, training_mask_dir, validation_img_dir, validation_mask_dir = [
    Path(folder_dict[folder]) for folder in folder_name
]

# %% [markdown]
# ### Select Training or Validation

# %%
# Choose which data you are checking "training" or "validation"
data_type = "training"

# %%
img_files, mask_files = get_file_paths(data_type)

# %% [markdown]
# ## Explore files <a name="explore"></a>

# %% [markdown]
# ##### List img files in img_dir

# %%
img_file_names = [file.name for file in img_files]
img_file_names

# %% [markdown]
# ##### List mask files in mask_dir

# %%
mask_file_names = [file.name for file in mask_files]
mask_file_names

# %% [markdown]
# ##### Check same number of img and mask files present

# %%
check_same_number_of_files_present(img_files, mask_files)

# %% [markdown]
# ## File cleaning <a name="filecleaning"></a>

# %% [markdown]
# ##### Change all file names to lower case

# %%
# Lower case img file names
img_files_lower = change_to_lower_case(img_files)

# Lower case mask file names
mask_files_lower = change_to_lower_case(mask_files)

# %% [markdown]
# ###### Check each mask file has corresponding img file & each img file has corresponding mask file

# %%
# Check each mask file has corresponding img file
vice_versa_check_mask_file_for_img_file(
    img_files_lower, mask_files_lower, for_mask_or_img="mask"
)

# %%
# Check each img file has corresponding mask file
vice_versa_check_mask_file_for_img_file(
    img_files_lower, mask_files_lower, for_mask_or_img="img"
)

# %% [markdown]
# ##### Check to ensure naming convention held for masks and img files

# %%
check_naming_convention_upheld(img_files_lower, mask_files_lower, data_type)

# %% [markdown]
# ## Mask file cleaning <a name="maskfilecleaning"></a>

# %%
# Clean data mask file
cleaning_of_mask_files(mask_files_lower, data_type)

# %% [markdown]
# # Checking complete. Now you can copy data files into Sharepoint GCP data ingest area

# %%
