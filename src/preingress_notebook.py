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
# # Pre GCP Ingress Notebook
#
# > Notebook to be run before any files are transferred to the SharePoint GCP ingress folder
#
# > You will need to manually copy contents of GCP_ingress folder on Sharepoint to your local machine and vice-versa
#
# > Do training and validation data **separately**
#
# > Only do **one pair** of mask and img files at a time
#
# > Remember to change **data_for** variable depending on whether training or validation is being checked

# %% [markdown]
# ## Set-up
#
# import re  # pattern matching
# import warnings  # used for sending warnings

# %%
# Load required libraries
from pathlib import Path  # working with file paths
import re  # pattern matching
import warnings  # used for sending warnings
import geopandas as gpd  # working with geospatial files and data
import numpy as np  # Used to ensure a training tile has more than one type

# Local imports
from functions_library import setup_sub_dir
from preingress_functions import (
    change_to_lower_case, 
    vice_versa_check_mask_file_for_img_file, 
    check_naming_convention_upheld, 
    cleaning_of_mask_files,
    check_same_number_of_files_present,
    create_path_list_variables
)

# %%
# Note directories of interest
data_dir = Path.cwd().parent.joinpath("data")
training_data_dir = data_dir.joinpath("training_data")
validation_data_dir = data_dir.joinpath("validation_data")

# Sub directories for training data
img_dir = setup_sub_dir(
    training_data_dir, "img"
)  # Note setup_sub_dir creates these if not present
mask_dir = setup_sub_dir(training_data_dir, "mask")

# Sub directories for validation data
validation_img_dir = setup_sub_dir(
    validation_data_dir, "img"
)  # Note setup_sub_dir creates these if not present
validation_mask_dir = setup_sub_dir(validation_data_dir, "mask")

# %% [markdown]
# ## Select Training or Validation

# %%
# Choose which data you are checking "training" or "validation"
data_for = "validation"

# %% [markdown]
# ### Explore files

# %%
# Create path lists for img and mask files

img_files, mask_files = create_path_list_variables(data_for, 
                                                   img_dir = img_dir, 
                                                   mask_dir = mask_dir, 
                                                   validation_img_dir = validation_img_dir, 
                                                   validation_mask_dir = validation_mask_dir)

# %% [markdown]
# ##### List img files in img folder 

# %%
img_file_names = [file.name for file in img_files]
img_file_names

# %% [markdown]
# ##### List mask files in mask folder

# %%
mask_file_names = [file.name for file in mask_files]
mask_file_names

# %% [markdown]
# ##### Check same number of img and mask files present

# %%
check_same_number_of_files_present(data_for, 
                                   img_dir = img_dir, 
                                   mask_dir = mask_dir, 
                                   validation_img_dir = validation_img_dir, 
                                   validation_mask_dir = validation_mask_dir)

# %% [markdown]
# ## General file cleaning
#
# * change all file names to lower case (see [`Path.rename()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename) and [`str.lower()`](https://www.programiz.com/python-programming/methods/string/lower)
# * check there each img file has corresponding mask file and _vice versa_ - both img and mask files should have same name except suffix
# * ensure naming convention upheld for tif and geojson? Should be: `training_data_<area>_<tile no>_<initials>_<bgr>.tif`
# * specific for img files! check banding? Check in with Laurence on this and see: https://github.com/datasciencecampus/somalia_unfpa_census_support/issues/173

# %% [markdown]
# ##### Change all training data file names to lower case

# %%
# Lower case img file names
img_files_lower = change_to_lower_case(img_files)

# Lower case mask file names
mask_files_lower = change_to_lower_case(mask_files)

# %% [markdown]
# ###### Check each mask file has corresponding img file & each img file has corresponding mask file

# %%
vice_versa_check_mask_file_for_img_file(
    img_files_lower, mask_files_lower, for_mask_or_img="mask"
)

# %%
vice_versa_check_mask_file_for_img_file(
    img_files_lower, mask_files_lower, for_mask_or_img="img"
)

# %% [markdown]
# ##### Check to ensure naming convention held for masks and img files

# %%
check_naming_convention_upheld(img_files_lower, mask_files_lower, data_for)

# %% [markdown]
# ## Mask file cleaning
#
# * check data in each geojson (see [reading geojson](https://docs.astraea.earth/hc/en-us/articles/360043919911-Read-a-GeoJSON-File-into-a-GeoPandas-DataFrame)):
#    * check there is a type column
#    * remove fid or id column
#    * check for na

# %%
# Clean data mask file
cleaning_of_mask_files(mask_files_lower)

# %% [markdown]
# # Checking complete remember to copy data files back into Sharepoint data ingest area once happy

# %%
