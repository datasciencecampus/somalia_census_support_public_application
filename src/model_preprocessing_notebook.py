# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# import standard and third party libraries
from pathlib import Path
import geopandas as gpd
import rasterio as rio
import numpy as np

# %%
# import custom functions
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
data_dir = Path.cwd().parent.joinpath("data")
planet_imgs_path = setup_sub_dir(data_dir, "planet_images")
priority_area_geojsons_dir = setup_sub_dir(data_dir, "priority_areas_geojson")

# %% [markdown]
# ### Preprocess Planet raster

# %%
raster_file_path = data_dir.joinpath("training_tile1.tif")

# %%
raster_filepath = raster_file_path
img_array = return_array_from_tiff(raster_filepath)
img_arr_reordered = change_band_order(img_array)
normalised_img = clip_and_normalize_raster(img_arr_reordered, 99)

# %% [markdown]
# ### Load DSC training data

# %%
training_data = gpd.read_file(data_dir.joinpath("training_data.geojson"))

# %% [markdown]
# ## Process training data into raster

# %%
building_class_list = ["House", "Tent", "Service"]

segmented_training_arr = rasterize_training_data(training_data, raster_file_path, building_class_list, "test3.tif")


# %%
def reorder_array(img_arr, height_index, width_index, bands_index):
    arr = np.transpose(img_arr, axes=[height_index, width_index, bands_index])
    return arr


# %%
# transpose array to (x, y, band) from (band, x, y)
normalised_img = reorder_array(normalised_img, 1, 2, 0)

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(segmented_training_arr)
plt.subplot(122)
plt.imshow(normalised_img[:,:,:3])
plt.show()

# %% [markdown]
# Output rasters as numpy binary objects. Need to be able to load in modelling environment without geospatial packages present (due to difficulties in installing these on Windows device). 

# %%
with open(data_dir.joinpath('normalised_sat_raster.npy'), 'wb') as f:
    np.save(f, normalised_img)

# %%
with open(data_dir.joinpath('training_mask_raster.npy'), 'wb') as f:
    np.save(f, segmented_training_arr)

# %%
