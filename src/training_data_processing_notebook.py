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
training_masks_dir = setup_sub_dir(data_dir, "training_masks")
priority_area_geojsons_dir = setup_sub_dir(data_dir, "priority_areas_geojson")

# %% [markdown]
# ### Load DSC training data

# %%
older_version_training_data = False

# %%
if older_version_training_data:
    training_data = gpd.read_file(data_dir.joinpath("training_data.geojson"))
    raster_file_path = data_dir.joinpath("training_tile1.tif")
else:
    doolow_training_data_dir= data_dir.joinpath("doolow_training_data")
    training_data = gpd.read_file(doolow_training_data_dir.joinpath("training_data.shp"))
    raster_file_path = doolow_training_data_dir.joinpath("doolow_training_raster.tif")
# TODO: remove the older version stuff when successful run through on newer training data succeeds

# %%
# Extra column not used and full of missing values
training_data = training_data.drop(columns=['fid'])
# Drop missing data as otherwise subseuqent rasterisation crashes
# TODO: check why missing data exists
training_data = training_data.dropna()
training_data.Type.value_counts()

# %% [markdown]
# ### Process training data into raster

# %%
building_class_list = ["House", "Tent", "Service"]

segmented_training_arr = rasterize_training_data(
    training_data,
    raster_file_path,
    building_class_list,
    training_masks_dir.joinpath(f"{raster_file_path.stem}_mask.tif")
)

# %% [markdown]
# ### Preprocess Planet raster

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
# ## Check training tile and training mask

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
