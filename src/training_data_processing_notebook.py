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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training data processing
#
# This is step 1 in the development of a training model. If all the training data is already processed (i.e. been through this process and then exported and saved as numpy binary objects) then you do not need to repeat this step.
#
# Ouputs are saved as numpy binary objects to be later handled in the modelling environment without geospatial packages.
#
# <br>
#
# <div class="warning" style='background-color:#e9d8fd; color: #69337a; border-left: solid #805ad5 4px; border-radius: 2px; padding:0.7em;'>
# <span>
#     <p style='margin-left:0.5em;'>
#         Currently only have 1 training tile from Doolow
#     </p></span>
#   </div>
#
#
# ## Contents
#
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Load training data](#loadtraining)
# 1. ##### [Training data to raster](#trainingraster)
# 1. ##### [Check training tile and raster](#checktrainingtile)
# 1. ##### [Output to numpy object](#outputnumpy)
# 1. ##### [Manipulate training tiles](#trainingmanipulation)
#

# %% [markdown]
# ## Set-up <a name="setup"></a>

# %% [markdown]
# ### Import libraries

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# %%
from functions_library import setup_sub_dir
from modelling_preprocessing import rasterize_training_data, reorder_array
from planet_img_processing_functions import (
    change_band_order,
    clip_and_normalize_raster,
    return_array_from_tiff,
)

# %% [markdown]
# ### Custom functions


# %% [markdown]
# ### Set-up filepaths

# %%
# set relevant filepaths
data_dir = Path.cwd().parent.joinpath("data")
planet_imgs_path = setup_sub_dir(data_dir, "planet_images")
training_masks_dir = setup_sub_dir(data_dir, "training_masks")
priority_area_geojsons_dir = setup_sub_dir(data_dir, "priority_areas_geojson")

# for outputting data into two folders (images and mask)
training_data_output_dir = setup_sub_dir(data_dir, "training_data_output")
img_dir = setup_sub_dir(training_data_output_dir, "img")
mask_dir = setup_sub_dir(training_data_output_dir, "mask")

# %%
# Doolow specific training data
# TODO: Adjust as more training data added from other areas?
doolow_training_data_dir = data_dir.joinpath("training_data_doolow")

# %% [markdown]
# ## Load DSC training data <a name="loadtraining"></a>

# %%
# load training polygons and raster
# TODO: Better system for loading in files when they exist
training_data = gpd.read_file(
    doolow_training_data_dir.joinpath("training_data_doolow_1.shp")
)
raster_file_path = doolow_training_data_dir.joinpath("training_data_doolow_1.tif")

# %%
# remove unused column
training_data = training_data.drop(columns=["fid"])

# check number of building type and no missing data
# if NA values then go back to QGIS to fix
training_data.Type.value_counts()

# %% [markdown]
# ## Training data to raster <a name="trainingraster"></a>

# %%
building_class_list = ["House", "Tent", "Service"]

segmented_training_arr = rasterize_training_data(
    training_data,
    raster_file_path,
    building_class_list,
    training_masks_dir.joinpath(f"{raster_file_path.stem}_mask.tif"),
)

# %% [markdown]
# ## Preprocess Planet raster <a name="planetrasterprocess"></a>

# %%
img_array = return_array_from_tiff(raster_file_path)
img_arr_reordered = change_band_order(img_array)
normalised_img = clip_and_normalize_raster(img_arr_reordered, 99)

# %%
# transpose array to (x, y, band) from (band, x, y)
normalised_img = reorder_array(normalised_img, 1, 2, 0)

# %% [markdown]
# ## Check training tile and training mask <a name="checktrainingtile"></a>

# %%
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(normalised_img[:, :, :3])
plt.subplot(122)
plt.imshow(segmented_training_arr)
plt.show()

# %% [markdown]
# ## Output rasters to numpy <a name="outputnumpy"></a>
#
# >Need to load in data to modelling environment that has no geospatial packages present so converting to numpy binary objects.

# %%
with open(img_dir.joinpath("d1_normalised_sat_raster.npy"), "wb") as f:
    np.save(f, normalised_img[:, :, :3])

# %%
with open(mask_dir.joinpath("d1_training_mask_raster.npy"), "wb") as f:
    np.save(f, segmented_training_arr)
