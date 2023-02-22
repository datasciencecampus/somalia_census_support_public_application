# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training data processing
#
# This notebook will prepare training data for input into the U-Net model. Ouputs are saved as numpy binary objects to be later handled in the modelling environment without geospatial packages.
#
#
# <div class="warning" style='background-color:#e9d8fd; color: #69337a; border-left: solid #805ad5 4px; border-radius: 2px; padding:0.7em;'>
# <span>
#     <p style='margin-left:0.5em;'>
#         Currently only have 2 training tiles both from Doolow
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
#

# %% [markdown]
# ## Set-up <a name="setup"></a>

# %% [markdown]
# ### Import libraries

# %%
from pathlib import Path
import geopandas as gpd
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### Custom functions

# %%
from functions_library import (
    setup_sub_dir
)

from planet_img_processing_functions import (
    return_array_from_tiff,
    change_band_order,
    clip_and_normalize_raster,
)

from modelling_preprocessing import rasterize_training_data

# %% [markdown]
# ### Set-up filepaths

# %%
# TODO: Add to functions library?
data_dir = Path.cwd().parent.joinpath("data")
planet_imgs_path = setup_sub_dir(data_dir, "planet_images")
training_masks_dir = setup_sub_dir(data_dir, "training_masks")
priority_area_geojsons_dir = setup_sub_dir(data_dir, "priority_areas_geojson")
training_data_numpy_dir = setup_sub_dir(data_dir, "training_data_numpy")

# %%
# Doolow specific training data
# TODO: Adjust as more training data added from other areas
doolow_training_data_dir= data_dir.joinpath("Training_Data_Doolow")

# %% [markdown]
# ## Load DSC training data <a name="loadtraining"></a>

# %%
# load training polygons and raster
# TODO: Better system for loading in files - or will this not matter?
training_data = gpd.read_file(doolow_training_data_dir.joinpath("training_data_doolow_1.shp"))
raster_file_path = doolow_training_data_dir.joinpath("training_data_doolow_1.tif")

# %%
# remove unused column
training_data = training_data.drop(columns=['fid'])

# check number of building type and no missing data
# TODO: Can reference against QGIS outputs to ensure same value - overkill?
training_data.Type.value_counts()

# %% [markdown]
# ## Training data to raster <a name="trainingraster"></a>

# %%
building_class_list = ["House", "Tent", "Service"]

segmented_training_arr = rasterize_training_data(
    training_data,
    raster_file_path,
    building_class_list,
    training_masks_dir.joinpath(f"{raster_file_path.stem}_mask.tif")
)

# %% [markdown]
# ## Preprocess Planet raster <a name="planetrasterprocess"></a>

# %%
img_array = return_array_from_tiff(raster_file_path)
img_arr_reordered = change_band_order(img_array)
normalised_img = clip_and_normalize_raster(img_arr_reordered, 99)


# %%
def reorder_array(img_arr, height_index, width_index, bands_index):
    # Re-order the array into height, width, bands order.
    arr = np.transpose(img_arr, axes=[height_index, width_index, bands_index])
    return arr


# %%
# transpose array to (x, y, band) from (band, x, y)
normalised_img = reorder_array(normalised_img, 1, 2, 0)

# %% [markdown]
# ## Check training tile and training mask <a name="checktrainingtile"></a>

# %%
#TODO: Clipping by extent in QGIS is overextending raster box - check impact
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(segmented_training_arr)
plt.subplot(122)
plt.imshow(normalised_img[:,:,:3])
plt.show()

# %% [markdown]
# ## Output rasters to numpy <a name="outputnumpy"></a>
#
# >Need to load in data to modelling environment that has no geospatial packages present so converting to numpy binary objects.

# %%
# TODO : Better system for saving file names for different training tiles/areas currently using dx (doolow 1/2)
with open(training_data_numpy_dir.joinpath('normalised_sat_raster_d1.npy'), 'wb') as f:
    np.save(f, normalised_img)

# %%
with open(training_data_numpy_dir.joinpath('training_mask_raster_d1.npy'), 'wb') as f:
    np.save(f, segmented_training_arr)
