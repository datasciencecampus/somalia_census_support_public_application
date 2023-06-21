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
# # Feasibility study - Pre-processing training data
#
# This notebook performs the geospatial processing of training images and masks and outputs as `.npy` arrays for input into the modelling notebook. This notebook only has to be run when new data is ingressed to GCP.
#
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: #31708f; background-color: #d9edf7; border-color: #bce8f1;">
# Before running this project ensure that the correct kernel is selected (top right). The default project environment name is `venv-somalia-gcp`.
# </div>
#
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

import json
import matplotlib.pyplot as plt
import numpy as np

from image_processing_functions import (
    change_band_order,
    check_img_files,
    clip_and_normalize_raster,
    reorder_array,
    return_array_from_tiff,
)
from mask_processing_functions import (
    check_mask_files,
    training_data_summary,
    process_geojson_files,
)

# %% [markdown]
# ### Set-up directories

# %%
# set data directory
data_dir = Path.cwd().parent.joinpath("data")

# set training_data directory within data folder
training_data_dir = data_dir.joinpath("training_data")

# set img and mask directories within training_data directory
img_dir = training_data_dir.joinpath("img")
mask_dir = training_data_dir.joinpath("mask")

# %% [markdown]
# ### Set image size
#
# U-Net architecture uses max pooling to downsample images across 4 levels, so we need to work with tiles that are divisable by 4 x 2.
# Training data tiles created in QGIS as ~200m x 200m (which equates to ~400 x 400 pixels as resolution is 0.5m/px). Tiles are cropped to 384 pixels (or 192m) as it is easier to crop than be completely accurate in QGIS.

# %%
img_size = 384

# %% [markdown]
# ## Image files <a name="images"></a>
#
# Reading in all `.tif` files in the `img_dir` then performing geospatial processing on them using functions from the `image_processing_functions.py` and saving outputted files as `.npy` arrays into the same folder.

# %%
# list all .tif files in directoy
img_files = list(img_dir.glob("*.tif"))

for img_file in img_files:
    # reading in file with rasterio
    img_array = return_array_from_tiff(img_file)

    # reorder bands
    arr_reordered = change_band_order(img_array)

    # clip to percentile
    arr_normalised = clip_and_normalize_raster(arr_reordered, 99)

    # reorder into height, width, band order
    arr_normalised = reorder_array(arr_normalised, 1, 2, 0)

    # re-sizing to img_size (defined above as 384)
    arr_normalised = arr_normalised[0:img_size, 0:img_size, :]

    # create a new filename without bgr
    img_filename = Path(img_file).stem.replace("_bgr", "").replace("_rgb", "")

    # save the NumPy array
    np.save(img_dir.joinpath(f"{img_filename}.npy"), arr_normalised)

# %%
# checking shape of .npy files matches
check_img_files(img_dir)

# %% [markdown]
# ## Mask files <a name="masks"></a>
#
# Reading in all `.GeoJSON` files in the `mask_dir`, matching files to corresponding `img`, performing geospatial processing with `mask_processing_functions.py`, and saving outputted files as `.npy` arrays into the same folder.
#
# Currently only using 'building' and 'tent' as classes - but may incorporate 'service' at a later stage, which is in the commented out code.

# %%
building_class_list = ["Building", "Tent"]

# %%
features_dict = process_geojson_files(mask_dir, img_dir, building_class_list, img_size)

# %%
# Output the completed features dictionary to a JSON for use in outputs notebook
output_file = mask_dir.joinpath("feature_dict.json")
with open(output_file, "w") as f:
    json.dump(features_dict, f, indent=4)

# %%
# checking shape of .npy files matches
check_mask_files(mask_dir)

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
