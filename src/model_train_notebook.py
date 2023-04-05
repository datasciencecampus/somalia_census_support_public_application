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
#     display_name: venv-somalia-gcp
#     language: python
#     name: venv-somalia-gcp
# ---

# %% [markdown]
# # Feasibility study - U-Net training model
#
# This notebook is looking at the feasibility of applying the U-Net architecture to identify formal and in-formal building structures in IDP camps in areas of interest in Somalia.
#
# While the overall aim is to apply the model across Somalia, this feasibility study focuses on 5 areas of interest. These areas of interest were the subject of a recent SNBS survey, and so, provide a unique opportunity to add some element of ground-truthed data to the model.
#
# ## Contents
#
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Load training data](#loadraster)
# 1. ##### [Data Augmentation](#dataaug)
# 1. ##### [Training parameters](#trainingparameters)
# 1. ##### [Format data for model input](#formatdata)
# 1. ##### [Outputs for visual checking](#output)

# %% [markdown]
# > **Note**
# > Before running this project ensure that the correct kernel is selected (top right). The default project environment name is `venv-somalia-gcp`.

# %% [markdown]
# ## Set-up <a name="setup"></a>

# %% [markdown]
# ### segmentation_models work-around

# %%
# since this model was built segmentation models has been updated to use tf.keras -
# recommended work around is to set env var as below

# %env SM_FRAMEWORK = tf.keras

# %% [markdown]
# ### Import libraries

import os
import random
from pathlib import Path

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# %%
from functions_library import setup_sub_dir
from modelling_preprocessing import rasterize_training_data, reorder_array
from multi_class_unet_model_build import jacard_coef, multi_unet_model
from planet_img_processing_functions import (
    change_band_order,
    clip_and_normalize_raster,
    return_array_from_tiff,
)

# %% [markdown]
# ### Import custom functions


# %% [markdown]
# ### Set-up filepaths

# %%
data_dir = Path.cwd().parent.joinpath("data")

training_data_dir = data_dir.joinpath("training_data")
img_dir = training_data_dir.joinpath("img")
mask_dir = training_data_dir.joinpath("mask")

# %% [markdown]
# ## Load training data <a name="loadraster"></a>

# %% [markdown]
# ### Image rasters

# %%
raster_dataset = []
for path, subdirs, files in os.walk(training_data_dir):

    dirname = path.split(os.path.sep)[-1]
    if dirname == "img":
        images = os.listdir(path)

        for i, image_name in enumerate(images):
            if image_name.endswith(".tif"):

                img_array = return_array_from_tiff(img_dir.joinpath(image_name))
                img_arr_reordered = change_band_order(img_array)
                normalised_img = clip_and_normalize_raster(img_arr_reordered, 99)
                normalised_img = reorder_array(normalised_img, 1, 2, 0)

                raster_dataset.append(normalised_img)

# %%
raster_dataset = np.array(raster_dataset)

# %% [markdown]
# ### Mask layers

# %%
mask_dataset = []

building_class_list = ["House", "Tent", "Service"]

for path, subdirs, files in os.walk(training_data_dir):

    dirname = path.split(os.path.sep)[-1]
    if dirname == "mask":
        masks = os.listdir(path)

        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".shp"):

                mask_filename = Path(mask_name).stem
                training_data = gpd.read_file(mask_dir.joinpath(mask_name))
                training_data = training_data.drop(columns=["fid"])

                segmented_training_arr = rasterize_training_data(
                    training_data,
                    img_dir.joinpath(image_name),
                    building_class_list,
                    mask_dir.joinpath(f"{mask_filename}.tif"),
                )

                mask_dataset.append(segmented_training_arr)


# %%
mask_dataset = np.array(mask_dataset)

# %%
# show random images for checking

image_number = random.randint(0, len(raster_dataset))

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(raster_dataset[image_number])
plt.subplot(122)
plt.imshow(mask_dataset[image_number])
plt.show()

# %% [markdown]
# ### Old workflow for single images that can't yet be deleted!

# %%
# opening single files - old version but works so keeping until multi file version working


training_data = gpd.read_file(mask_dir.joinpath("training_data_doolow_1.shp"))
raster_file_path = img_dir.joinpath("training_data_doolow_1.tif")


# remove unused column
training_data = training_data.drop(columns=["fid"])

building_class_list = ["House", "Tent", "Service"]

segmented_training_arr = rasterize_training_data(
    training_data,
    raster_file_path,
    building_class_list,
    mask_dir.joinpath(f"{raster_file_path.stem}_mask.tif"),
)

img_array = return_array_from_tiff(raster_file_path)
img_arr_reordered = change_band_order(img_array)
normalised_img = clip_and_normalize_raster(img_arr_reordered, 99)

# transpose array to (x, y, band) from (band, x, y)
normalised_img = reorder_array(normalised_img, 1, 2, 0)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(normalised_img[:, :, :3])
plt.subplot(122)
plt.imshow(segmented_training_arr)
plt.show()


# with open(img_dir.joinpath("d1_normalised_sat_raster.npy"), "rb") as f:
# normalised_sat_raster = np.load(f)

# normalised_sat_raster_uncropped = normalised_sat_raster

# with open(mask_dir.joinpath("d1_training_mask_raster.npy"), "rb") as f:
# training_mask_raster = np.load(f)

# %%
img_size = 574

normalised_sat_raster = normalised_img[0:img_size, 0:img_size, :]
normalised_sat_raster.shape

# %%
training_mask_raster = segmented_training_arr[0:img_size, 0:img_size]
training_mask_raster.shape

# %% [markdown]
# ## Data Augmentation <a name="dataaug"></a>
#

# %% [markdown]
# ### Image scaling
#
#
# U-Net architecture uses max pooling to downsample images across 4 levels, so we need to work with tiles that are divisable by 4 x 2.
#
# Training data tiles created in QGIS as ~200 x 200 (which equates to ~400 x 400 as resolution is 0.5m/px). Intention is for tiles to be cropped to 192 (or should it be 384?)

# %%
# img_size = 384

# %% [markdown]
# ### Image manipulation

# %%
# Not tested!

# def augment(input_image, input_mask):
# if tf.random.uniform(()) > 0.5:
# input_image = tf.image.flip_left_right(input_image)
# input_mask = tf.image.flip_left_right(input_mask)
# return input_image, input_mask

# not finished
# def augment(input_image, input_mask):

# flip vertically
# input_image = np.flipud(normalised_sat_raster)
# input_mask = np.flipud(training_mask_raster)

# flip horizontal
# input_image = np.fliplr(normalised_sat_raster)
# input_mask = np.fliplr(training_mask_raster)

# rotate 90 degrees
# input_image = np.rot90(normalised_sat_raster)
# input_mask = np.rot90(training_mask_raster)

# return input_image, input_mask

# %% [markdown]
# ## Training parameters <a name="trainingparameters"></a>

# %%
img_height, img_width, num_channels = normalised_sat_raster.shape

# %% [markdown]
# ### Weights
#
# The reltaive weights between building classes is one parameter that can be tweaked and optimised.

# %%
# Parameters for model
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss

# Calaculates the relative frequency of each class within the lablled mask.
weights = compute_class_weight(
    "balanced",
    classes=np.unique(training_mask_raster),
    y=np.ravel(training_mask_raster, order="C"),
)

# Alternatively, could try balanced weights between classes:
# weights = [0.25, 0.25, 0.25, 0.25]

print(weights)


# %% [markdown]
# ### Loss function
# The loss function is an additional parameter that can be tweaked and optimised.

# %%
# create custom loss function for model training
# this is inspired from the tutorial used to create this initial code
# TODO: Explore alternatives
dice_loss = sm.losses.DiceLoss(class_weights=weights)  # corrects for class imbalance
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# %% [markdown]
# ### Metrics
# The metrics used to measure the model performance can be optimised also.

# %%
metrics = ["accuracy", jacard_coef]

# %% [markdown]
# ## Get data into format the model expects <a name="formatdata"></a>

# %%
n_classes = len(np.unique(training_mask_raster))
n_classes

# %%
# one-hot encode building classes in training mask
labels_categorical = to_categorical(training_mask_raster, num_classes=n_classes)

# duplicates the single training mask to simulate the stack of training data
# that will exist at some stage
# TODO: Remove this later!
labels_categorical = np.repeat(labels_categorical[..., None], 5, axis=3)

# reorder the array to image list index, height, width, categorical class
labels_categorical = np.transpose(labels_categorical, axes=[3, 0, 1, 2])

labels_categorical.shape

# %%
# duplicates the single training image to simulate the stack of training data
# that will exist at some stage
# TODO: Add manipulated training data
# TODO: Remove this later!
stacked_training_rasters = np.repeat(normalised_sat_raster[..., None], 5, axis=3)

# reorder the array to image list index, height, width, categorical class
stacked_training_rasters = np.transpose(stacked_training_rasters, axes=[3, 0, 1, 2])

stacked_training_rasters.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(
    stacked_training_rasters, labels_categorical, test_size=0.20, random_state=42
)


# %%
def get_model():
    return multi_unet_model(
        n_classes=n_classes,
        IMG_HEIGHT=img_height,
        IMG_WIDTH=img_width,
        IMG_CHANNELS=num_channels,
    )


# %%
model = get_model()

model.compile(optimizer="adam", loss=total_loss, metrics=metrics)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

model.summary()

# how many times the model runs through training data
# use with callbacks to find the optimum number
num_epochs = 50

# early stopping monitors the model to prevent under/over fitting by running too few/many epochs
# monitors validation loss, with a set patience (i.e. if the model thinks it has found the right number
# of epochs then it runs a few more to check it was correct)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor="val_loss"),
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
]


history1 = model.fit(
    X_train,
    y_train,
    batch_size=16,
    verbose=1,
    epochs=num_epochs,
    validation_data=(X_test, y_test),
    shuffle=False,
    callbacks=callbacks,
)

# %%
# %load_ext tensorboard
# %tensorboard --logdir logs/

# %%
models_dir = setup_sub_dir(Path.cwd().parent, "models")
model.save(
    models_dir.joinpath(f"trail_run_{num_epochs}epochs_{img_size}pix_doolow.hdf5")
)

# %%
y_pred = model.predict(X_test)

predicted_img = np.argmax(y_pred, axis=3)[0, :, :]

# %% [markdown]
# ## Output visual checking <a name="output"></a>

# %%
with open(data_dir.joinpath("pred.npy"), "rb") as f:
    predicted_img = np.load(f)

img_size = predicted_img.shape[1]

# %%
with open(mask_dir.joinpath("d1_training_mask_raster.npy"), "rb") as f:
    training_mask_raster = np.load(f)

# Crop to size of modelling tile
mask = training_mask_raster[0:img_size, 0:img_size]
mask.shape

# %%
with open(img_dir.joinpath("d1_normalised_sat_raster.npy"), "rb") as f:
    normalised_sat_raster = np.load(f)

# Crop to size of modelling tile
normalised_sat_raster = normalised_sat_raster[0:img_size, 0:img_size, :]
normalised_sat_raster.shape

# %%
plt.figure(figsize=(13, 8))
plt.subplot(231)
plt.title("Image")
plt.imshow(normalised_sat_raster[:, :, :3])
plt.subplot(232)
plt.title("Mask")
plt.imshow(mask[:, :])
plt.subplot(233)
plt.title("Prediction of classes")
plt.imshow(predicted_img[0, :, :, 0])
plt.subplot(234)
plt.title("Prediction of classes")
plt.imshow(predicted_img[0, :, :, 1])
plt.subplot(235)
plt.title("Prediction of classes")
plt.imshow(predicted_img[0, :, :, 2])
plt.subplot(236)
plt.title("Prediction of classes")
plt.imshow(predicted_img[0, :, :, 3])
plt.show()

# %%
