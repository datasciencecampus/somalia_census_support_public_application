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
import folium
import json
from pathlib import Path
import geopandas as gpd
import rasterio as rio
import numpy as np

# %%
# import custom functions
from functions_library import (
    setup_sub_dir,
    list_directories_at_path
)

from planet_img_processing_functions import (
    check_zipped_dirs_and_unzip,
    extract_dates_from_image_filenames,
    get_raster_list_for_given_area,
    return_array_from_tiff,
    change_band_order,
    clip_and_normalize_raster,
)

from geospatial_util_functions import (
    convert_shapefile_to_geojson,
    get_reprojected_bounds,
    check_crs_and_reset
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


# %%
def expand_dimensions(training_mask_arr, reference_tif):
    """
    Model expects same dimensions in satellite raster and training mask. So expand training
    array to match shape of image by duplicating single channel.
    
    UNCLEAR IF NEEDED - DELETE if not
    """
    num_bands = reference_tif.shape[0]
    expanded_arr = np.repeat(training_mask_arr[...,None], num_bands, axis = 2)
    transposed_expanded_arr = np.transpose(expanded_arr, axes=[2, 0, 1])
    return(transposed_expanded_arr)


# %%
training_mask = expand_dimensions(segmented_training_arr, normalised_img)

# %%
num_channels, img_height, img_width = normalised_img.shape 

# %%
import segmentation_models as sm

# %%
# keras installation requires tensoreflow installed in back
from multi_class_unet_model_build import multi_unet_model, jacard_coef

# %%
#Parameters for model
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(
    'balanced',
    np.unique(np.ravel(segmented_training_arr,order='C')),
    np.ravel(segmented_training_arr,order='C')
)
print(weights)

# %%
n_classes = len(np.unique(segmented_training_arr))
n_classes

# %%
from keras.utils import to_categorical
# one-hot encode building classes in training mask
labels_categorical = to_categorical(segmented_training_arr, num_classes=n_classes)
labels_categorical.shape

# %%
# create custom loss function for model training
dice_loss = sm.losses.DiceLoss(class_weights=weights) # corrects for class imbalance
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

# %%
labels_categorical = np.repeat(labels_categorical[...,None], 7, axis = 3)
labels_categorical.shape

# %%
labels_categorical = np.transpose(labels_categorical, axes = [3, 0, 1, 2])
labels_categorical.shape

# %%
#TODO: for each training tile, generate additional ones by rotating them and mirroring them
# - will need to do same for corresponding masks of course

#NOTE: These have to be order with list index, then x, y, band/class
stacked_training_rasters = np.repeat(normalised_img[...,None], 7, axis = 3)
stacked_training_rasters = np.transpose(stacked_training_rasters, axes = [3, 0, 1, 2])

stacked_training_rasters.shape

# %%
from sklearn.model_selection import train_test_split

# stacked_training_rasters = stack of individual training raster tiles of same dimensions

X_train, X_test, y_train, y_test = train_test_split(
    stacked_training_rasters,
    labels_categorical,
    test_size = 0.20,
    random_state = 42
    )

# %%
X_train.shape


# %%
def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=img_height, IMG_WIDTH=img_width, IMG_CHANNELS=num_channels)

model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()


history1 = model.fit(X_train,
                     y_train,
                     batch_size = 16,
                     verbose=1,
                     epochs=100,
                     validation_data=(X_test, y_test),
                     shuffle=False
                    )
