# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: venv-somalia-gcp
#     language: python
#     name: venv-somalia-gcp
# ---

# %% [markdown]
# # Pre-modelling processing
#
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: #31708f; background-color: #d9edf7; border-color: #bce8f1;">
# Before running this project ensure that the correct kernel is selected (top right). The default project environment name is `venv-somalia-gcp`.
# </div>
#
# #### Purpose
# Processes locally stored img and mask files and outputs as `.npy`, which are saved in the same folder location.
#
# #### Things to note
# * Only has to be run if `download_data_from_ingress` has been run - as `.npy` files are saved
# * Check kernel
#
#
# ## Contents
#
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Image files](#images)
# 1. ##### [Mask files](#masks)
# 1. ##### [Training data summary](#trainingsummary)
# 1. ##### [Visual checking - images](#imagevisual)
# 1. ##### [Visual checking - masks](#maskvisual)

# %% [markdown]
# ## Set up <a name="setup"></a>

# %% [markdown]
# ### Import libraries & functions

# %%
# import libraries
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

from functions_library import get_data_paths

from image_processing_functions import (
    remove_bgr_from_filename,
    check_img_files,
    process_image,
)
from mask_processing_functions import (
    rasterize_training_data,
    training_data_summary,
)

# %% [markdown]
# ### Set-up directories

# %%
# set data directory
data_dir = Path.cwd().parent.joinpath("data")

# get all sub directories within data forlder
sub_dir = [subdir.name for subdir in data_dir.iterdir() if subdir.is_dir()]

# %% [markdown]
# ### Select sub directory

# %%
folder_dropdown = widgets.Dropdown(options=sub_dir, description="select folder:")
display(folder_dropdown)

# %%
# set img and mask directories based on seelcted folder above
img_dir, mask_dir = get_data_paths(folder_dropdown.value)
print(img_dir)
print(mask_dir)

# %% [markdown]
# ### Set image size
#
# U-Net architecture uses max pooling to downsample images across 4 levels, so we need to work with tiles that are divisable by 4 x 2.
# Training data tiles created in QGIS as ~200m x 200m (which equates to ~400 x 400 pixels as resolution is 0.5m/px). Tiles are cropped to 384 pixels (or 192m) as it is easier to crop than be completely accurate in QGIS.

# %%
img_size = 256

# %% [markdown]
# ## Image files <a name="images"></a>
#
# Reading in all `.tif` files in the `img_dir` then performing geospatial processing on them using functions from the `image_processing_functions.py` and saving outputted files as `.npy` arrays into the same folder.

# %%
# list all .tif files in directoy
img_files = list(img_dir.glob("*.tif"))

# %%
# process .geotiff and save as .npy
for img_file in img_files:
    process_image(img_file, img_size, img_dir)

# %%
# checking shape of .npy files matches
check_img_files(img_dir, (256, 256, 4))

# %%
# remove _bgr from file names if present
remove_bgr_from_filename(img_dir, img_files)

# %% [markdown]
# ## Mask files <a name="masks"></a>
#
# Reading in all `.GeoJSON` files in the `mask_dir`, matching files to corresponding `img`, performing geospatial processing with `mask_processing_functions.py`, and saving outputted files as `.npy` arrays into the same folder.
#
# Currently only using 'building' and 'tent' as classes - but may incorporate 'service' at a later stage, which is in the commented out code.

# %%
building_class_list = ["building", "tent"]

# %%
features_dict = {}

# %%
# loop through the GeoJSON files
for mask_path in mask_dir.glob("*.geojson"):
    rasterize_training_data(
        mask_path, mask_dir, img_dir, building_class_list, img_size, features_dict
    )

# %% [markdown]
# ## Training data summary<a name="trainingsummary"></a>
#
# This section is duplicating work from above but it joins all mask files together into a geopandas data frame to quickly overview data.

# %%
# joining masks together to count building types
training_data, value_counts, structure_stats = training_data_summary(mask_dir)

# %%
# building types
value_counts

# %%
# structure stats
structure_stats

# %%
# check pre-ingress worked - if type = true then workflow won't work

area_empty = training_data["Area"].isna().any()
type_empty = training_data["Type"].isna().any()

print("Is area column empty?", area_empty)
print("Is type column empty?", type_empty)

# %% [markdown]
# ## Visual checking - images <a name="imagevisual"></a>
#
#

# %%
# finding all .npy files - those converted above
file_list = [f for f in img_dir.glob("*.npy") if np.load(f).shape[-1] == 4]

# read in .npy files
image_list = [np.load(f) for f in file_list]

# plot the images
# display a maximum of 16 images
for i in range(min(16, len(image_list))):
    img = image_list[i]

    # create a 4 x 4 grid
    plt.subplot(4, 4, i + 1)

    # normalise the data to the range of 0 to 1
    img_normalised = img.astype(np.float32) / np.max(img)

    # show the first 3 channels (RGB)
    plt.imshow(img_normalised[..., :3])

    # plt.title(file_list[i].name) # use file name as title
    plt.axis("off")
plt.show()

# %% [markdown]
# ## Visual checking - masks <a name="maskvisual"></a>

# %%
# finding all .npy files - those converted above
file_list = [f for f in mask_dir.glob("*.npy")]

# read in .npy files
mask_list = [np.load(f) for f in file_list]

# plot the masks
# display a maximum of 16 images
for i in range(min(16, len(mask_list))):
    mask = mask_list[i]

    # create a 4 x 4 grid
    plt.subplot(4, 4, i + 1)

    # plot the mask
    plt.imshow(mask)

    # plt.title(file_list[i].name) # use file name as title
    plt.axis("off")
plt.show()

# %%
