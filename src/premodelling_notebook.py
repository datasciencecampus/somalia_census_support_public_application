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
# This notebook does the geospatial processing of training images and masks and outputs as `.npy` arrays for input into the modelling notebook.
#
# ## Contents
#
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Image files](#images)
# 1. ##### [Mask files](#masks)

# %% [markdown]
# ## Set up <a name="setup"></a>

# %% [markdown]
# ### Import libraries & functions

# %%
# import libraries

from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image

from modelling_preprocessing import rasterize_training_data, reorder_array
from planet_img_processing_functions import change_band_order, clip_and_normalize_raster

# %%
# import custom functions

# TODO: MOVE reorder_array into planet_img_processing_functions
# TODO: CLEAN plant_img_processing_functions
# TODO: RATIONALISE functions into one script?


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
# ## Image files <a name="images"></a>

# %%
# list all .tif files in directoy
tif_files = list(img_dir.glob("*.tif"))

for tif_file in tif_files:

    # open the tif file
    with Image.open(tif_file) as im:

        # convert image to a NumPy array
        arr = np.array(im)

        # reorder bands
        arr_reordered = change_band_order(arr)

        # clip to percentile
        arr_normalised = clip_and_normalize_raster(arr_reordered, 99)

        # reorder into height, width, band order
        arr_normalised = reorder_array(arr_normalised, 1, 2, 0)

        # create a new filename
        npy_file = img_dir / (tif_file.stem)

        # save the NumPy array
        np.save(npy_file, arr_normalised)

# %% [markdown]
# ## Mask files <a name="masks"></a>

# %%
building_class_list = ["Building", "Tent"]

# %%
# loop through the GeoJSON files
for mask_path in mask_dir.glob("*.geojson"):

    # load the GeoJSON into a GeoPandas dataframe
    mask_gdf = gpd.read_file(mask_path)

    # add a 'Type' column if it doesn't exist (should be background tiles only)
    if "Type" not in mask_gdf.columns:
        mask_gdf["Type"] = ""

    # replace values in 'Type' column
    mask_gdf["Type"].replace({"House": "Building", "Service": "Building"}, inplace=True)

    # define corresponding image filename
    mask_filename = Path(mask_path).stem
    image_filename = f"{mask_filename}_bgr.tif"
    image_file = img_dir.joinpath(image_filename)

    # if the geometry is empty (background tiles)
    if mask_gdf.geometry.is_empty.all():

        # create a coresponding NumPy array of zeros
        zeros_array = np.zeros((mask_gdf.shape[0], 1))
        # create a new filename
        npy_mask_file = mask_dir / (mask_filename)

        np.save(f"{npy_mask_file}_mask.npy", zeros_array)
    else:

        # create rasterized training image
        segmented_training_arr = rasterize_training_data(
            mask_gdf,
            image_file,
            building_class_list,
            mask_dir.joinpath(f"{mask_filename}_mask.tif"),
        )

        # save the NumPy array
        np.save(f"{npy_mask_file}_mask.npy", segmented_training_arr)
