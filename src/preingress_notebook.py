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
    cleaning_of_mask_files
)

# %%
# Note directories of interest
data_dir = Path.cwd().parent.joinpath("data")
training_data_dir = data_dir.joinpath("training_data")
img_dir = setup_sub_dir(
    training_data_dir, "img"
)  # Note setup_sub_dir creates these if not present
mask_dir = setup_sub_dir(training_data_dir, "mask")

# %% [markdown]
# ## Explore files

# %%
# Get all the img and mask files present

# Absolute path for img files
img_files = list(img_dir.glob("*.tif"))

# Absolute path for mask files
mask_files = list(mask_dir.glob("*.geojson"))

# Check that same number of imgs and mask files present - if not then warning
if len(img_files) != len(mask_files):
    warnings.warn(
        f"Number of image files {len(img_files)} doesn't match number of mask files {len(mask_files)}"
    )

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
# ## General file cleaning
#
# * change all file names to lower case (see [`Path.rename()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename) and [`str.lower()`](https://www.programiz.com/python-programming/methods/string/lower)
# * check there each img file has corresponding mask file and _vice versa_ - both img and mask files should have same name except suffix
# * ensure naming convention upheld for tif and geojson? Should be: `training_data_<area>_<tile no>_<initials>_<bgr>.tif`
# * specific for img files! check banding? Check in with Laurence on this and see: https://github.com/datasciencecampus/somalia_unfpa_census_support/issues/173

# %% [markdown]
# ##### Change all file names to lower case

# %%
# Lower case img file names
img_files_lower = change_to_lower_case(img_files)

# Lower case mask file names
mask_files_lower = change_to_lower_case(mask_files)

# %% [markdown]
# ###### Check each mask file has corresponding img file & check each img file has corresponding mask file

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
check_naming_convention_upheld(img_files_lower, mask_files_lower)

# %% [markdown]
# ## Mask file cleaning
#
# * check data in each geojson (see [reading geojson](https://docs.astraea.earth/hc/en-us/articles/360043919911-Read-a-GeoJSON-File-into-a-GeoPandas-DataFrame)):
#    * check there is a type column
#    * remove fid or id column
#    * check for na

# %%
cleaning_of_mask_files(mask_files_lower)

# %% [markdown]
# # Checking complete remember to copy data files back into Sharepoint data ingest area once happy

# %%
