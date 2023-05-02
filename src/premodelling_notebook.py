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
# This notebook is part of the feasibility study into applying the U-Net architecture to identify formal and in-formal building structures in IDP camps in areas of interest in Somalia.
#
# While the overall aim is to apply the model across all IDP camps in Somalia, this feasibility study focuses on 4 areas of interest:
# * Baidoa
# * Beledweyne
# * Kismayo
# * Mogadishu
#
# These areas of interest were the subject of a recent Somalia National Bureau of Statistics (SNBS) survey, and so, provide a unique opportunity to add some element of ground-truthed data to the model.
#
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: #31708f; background-color: #d9edf7; border-color: #bce8f1;">
# Before running this project ensure that the correct kernel is selected (top right). The default project environment name is `venv-somalia-gcp`.
# </div>
#
# This notebook performs the geospatial processing of training images and masks and outputs as `.npy` arrays for input into the modelling notebook. This notebook only has to be run once and when new training data is added.
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

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from modelling_preprocessing import rasterize_training_data, reorder_array
from planet_img_processing_functions import (
    change_band_order,
    clip_and_normalize_raster,
    return_array_from_tiff,
)

# %%
# import custom functions

# TODO: MOVE reorder_array into planet_img_processing_functions
# TODO: CLEAN plant_img_processing_functions
# TODO: RATIONALISE functions into one script?
# TODO: MOVE column functions out of rasterize_training_data and into function script with mask functions below?


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
# Reading in all `.tif` files in the `img_dir` then performing geospatial processing on them using functions from the `planet_processing_functions.py` and saving outputted files as `npy` arrays into the same folder

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
# checking all image arrays have the same shape

# list all .npy files in directory
img_file_list = img_dir.glob("*npy")

# the shape we want all files to have
ref_shape = (384, 384, 4)

for file in img_file_list:
    img_array = np.load(file)

    # checking each file compared to reference shape. Will return error if one doesn't match
    if img_array.shape != ref_shape:
        warnings.warn(f"{file} has a different shape than the reference shape")

# %% [markdown]
# ## Mask files <a name="masks"></a>
#
# Reading in all `.GeoJSON` files in the `mask_dir`, matching files to corresponding `img`, performing geospatial processing and saving outputted files as `npt` arrays into the same folder

# %%
building_class_list = ["Building", "Tent"]

# %%
# loop through the GeoJSON files
for mask_path in mask_dir.glob("*.geojson"):

    # load the GeoJSON into a GeoPandas dataframe
    mask_gdf = gpd.read_file(mask_path)

    # !!!!Temporary fix to sort the bad QGIS import!!!!
    # Remove after next Ingress with updated file.
    if mask_path.stem == "training_data_doolow_1_jo":
        mask_gdf.crs = 102100

    # add a 'Type' column if it doesn't exist (should be background tiles only)
    if "Type" not in mask_gdf.columns:
        mask_gdf["Type"] = ""

    # replace values in 'Type' column
    mask_gdf["Type"].replace({"House": "Building", "Service": "Building"}, inplace=True)

    # define corresponding image filename
    mask_filename = Path(mask_path).stem
    image_filename = f"{mask_filename}_bgr.tif"
    image_file = img_dir.joinpath(image_filename)
    # print(mask_filename)

    # create rasterized training image
    segmented_training_arr = rasterize_training_data(
        mask_gdf,
        image_file,
        building_class_list,
        mask_dir.joinpath(f"{mask_filename}.tif"),
    )

    # re-sizing to img_size (defined above as 384)
    normalised_training_arr = segmented_training_arr[0:img_size, 0:img_size]

    # save the NumPy array
    np.save(mask_dir.joinpath(f"{mask_filename}.npy"), normalised_training_arr)

# %%
# checking all mask arrays have the same shape

# list all files in directory
mask_file_list = mask_dir.glob("*npy")

# the shape we want each file to have
ref_shape = (384, 384)

for file in mask_file_list:
    mask_array = np.load(file)

    # returns an error if any of the files don't match reference shape
    if mask_array.shape != ref_shape:
        warnings.warn(f"{file} has a different shape than the reference shape")

# %% [markdown]
# ## Training data summary<a name="trainingsummary"></a>

# %%
# iterate over all the files in the directroy
for file in mask_dir.iterdir():

    # check for file extension .geojson
    if file.suffix == ".geojson":

        # open the file and read its contents
        training_data = gpd.read_file(file)

        # add a 'Type' column if it doesn't exist (should be background tiles only)
        if "Type" not in training_data.columns:
            training_data["Type"] = ""

        # replace values in 'Type' column
        training_data["Type"].replace(
            {"House": "Building", "Service": "Building"}, inplace=True
        )

# %%
# counts of type column
training_data.groupby("Type").size()

# %% [markdown]
# ## Visual checking - images <a name="imagevisual"></a>
#
#

# %%
import tifffile as tiff

# identifying .tif files with 4 channels
file_list = [f for f in img_dir.glob("*.tif") if tiff.imread(f).shape[-1] == 4]

# reading in .tif files
image_list = [tiff.imread(f) for f in file_list]

# plot the images
for i, img in enumerate(image_list):

    plt.subplot(4, 4, i + 1)  # create a 4 x 4 grid
    plt.imshow(img[..., :3])  # show the first 3 channels (RGB)
    # plt.title(file_list[i].name) # use file name as title
    plt.axis("off")  # axis off
plt.show()

# %%
# finding all .npy files - those converted above
file_list = [f for f in img_dir.glob("*.npy") if np.load(f).shape[-1] == 4]

# read in .npy files
image_list = [np.load(f) for f in file_list]

# plot the images
for i, img in enumerate(image_list):

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
print(file_list)

# plot the images
for i, mask in enumerate(mask_list):

    # create a 4 x 4 grid
    plt.subplot(4, 4, i + 1)

    # show the first 3 channels (RGB)
    plt.imshow(mask)

    # plt.title(file_list[i].name) # use file name as title
    plt.axis("off")
plt.show()

# %%

# %%

# %%
