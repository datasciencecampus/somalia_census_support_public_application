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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## To be used following the `model_train_notebook` process
#
# Note this is only really needed as a stand-alone notebook because I have been unable to install tensorflow and geospatial/data viz packages on the same environment, necessitating a purely tensorflow focussed notebook

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

# %% [markdown]
# ### Load prediceted data

# %%
with open(data_dir.joinpath('pred.npy'), 'rb') as f:
    pred = np.load(f)

# %% [markdown]
# ### Load original training image

# %%
with open(data_dir.joinpath('normalised_sat_raster.npy'), 'rb') as f:
    normalised_sat_raster = np.load(f)
    
img_size = pred.shape[1]
# Crop to size of modelling tile
normalised_sat_raster = normalised_sat_raster[0:img_size, 0:img_size, :]
normalised_sat_raster.shape

# %% [markdown]
# ### Load original training mask

# %%
with open(data_dir.joinpath('training_mask_raster.npy'), 'rb') as f:
    mask = np.load(f)
    
img_size = pred.shape[1]
# Crop to size of modelling tile
mask = mask[0:img_size, 0:img_size]
mask.shape

# %% [markdown]
# ## Compare model prediction with image

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(13, 8))
plt.subplot(231)
plt.title("Image")
plt.imshow(normalised_sat_raster[:,:,:3])
plt.subplot(232)
plt.title("Mask")
plt.imshow(mask[:,:])
plt.subplot(233)
plt.title("Prediction of classes")
plt.imshow(pred[0,:,:,0])
plt.subplot(234)
plt.title("Prediction of classes")
plt.imshow(pred[0,:,:,1])
plt.subplot(235)
plt.title("Prediction of classes")
plt.imshow(pred[0,:,:,2])
plt.subplot(236)
plt.title("Prediction of classes")
plt.imshow(pred[0,:,:,3])
plt.show()

# %%
