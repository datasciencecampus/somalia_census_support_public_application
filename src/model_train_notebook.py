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
# ## Contents
#
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Load training data](#loadraster)
# 1. ##### [Data Augmentation](#dataaug)
# 1. ##### [Training parameters](#trainingparameters)
# 1. ##### [Format data for model input](#formatdata)
# 1. ##### [Model](#model)
# 1. ##### [Outputs for visual checking](#output)

# %% [markdown]
# ## Set-up <a name="setup"></a>

# %% [markdown]
# ### segmentation_models framework

# %%
# choosing segmentation_models framework

# %env SM_FRAMEWORK = tf.keras

# %% [markdown]
# ### Import libraries & custom functions

# %%
import os
from pathlib import Path

# %%
import geopandas as gpd
import matplotlib as mpl
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
# ### Set-up filepaths

# %%
data_dir = Path.cwd().parent.joinpath("data")

training_data_dir = data_dir.joinpath("training_data")
img_dir = setup_sub_dir(training_data_dir, "img")
mask_dir = setup_sub_dir(training_data_dir, "mask")

models_dir = setup_sub_dir(Path.cwd().parent, "models")

# %% [markdown]
# ## Load training data <a name="loadraster"></a>

# %% [markdown]
# ### Image rasters

# %%
image_dataset = []
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

                image_dataset.append(normalised_img)

# %%
image_dataset = np.array(image_dataset, dtype=object)

# %% [markdown]
# ### Mask layers - NOT CURRENTLY WORKING

# %%
training_data = gpd.read_file(mask_dir.joinpath("training_data_baidoa_jo.shp"))
training_data["Type"].value_counts(dropna=False)

# %%
mask_dataset = []

building_class_list = ["Building", "Tent"]

for path, subdirs, files in os.walk(training_data_dir):
    dirname = path.split(os.path.sep)[-1]
    if dirname == "mask":
        masks = os.listdir(path)

        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".shp"):

                mask_filename = Path(mask_name).stem
                print(mask_filename)
                training_data = gpd.read_file(mask_dir.joinpath(mask_name))
                training_data = training_data["Type"].replace(
                    ["Service", "House"], "Building"
                )

                segmented_training_arr = rasterize_training_data(
                    training_data,
                    img_dir.joinpath(f"{mask_filename}.tif"),
                    building_class_list,
                    mask_dir.joinpath(f"{mask_filename}.tif"),
                )

            # mask_dataset.append(segmented_training_arr)

# %%
mask_dataset = np.array(mask_dataset, dtype=object)

# %% [markdown]
# ### Load training data - old system

# %%
test_1 = image_dataset[1]
test_2 = image_dataset[2]

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(test_1[:, :, :3])
plt.subplot(122)
plt.imshow(test_2[:, :, :3])
plt.show()

# %%
training_data = gpd.read_file(mask_dir.joinpath("training_data_beledweyne_LJ.shp"))
raster_file_path = img_dir.joinpath("training_data_beledweyne_LJ.tif")

# %%
training_data["Type"] = training_data["Type"].fillna(0)

building_class_list = [0]

# %%
# remove unused column
# training_data = training_data.drop(columns=["fid"])

# building_class_list = ["House", "Tent", "Service"]

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


# %%
n_classes = len(np.unique(segmented_training_arr))

# Create colour map for preservation at point of displaying classes
col_map = mpl.cm.get_cmap("viridis", n_classes)

# %%
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(normalised_img[:, :, :3])
plt.subplot(122)
plt.imshow(segmented_training_arr, cmap=col_map)
plt.show()

# %%
img_size = 576

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
# Training data tiles created in QGIS as ~200m x 200m (which equates to ~400 x 400 pixels as resolution is 0.5m/px). Intention is for tiles to be cropped to 384 pixels (or 192m)

# %%
# img_size = 384

# %% [markdown]
# ### Image manipulation

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
# Here we are using focal loss, an extension of cross-entropy loss. By default, focal loss reduces the weights of well-classified objects, those that have a probability >0.5, and increases the weights of objects with a probability of <0.5. In practice what this means is it reduces the weight (i.e. spends less time focusing on) easily classifable objects (i.e. service buildings) and instead focuses on hard to classify objects (i.e. tents).
#
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: #8a6d3b;; background-color: #fcf8e3; border-color: #faebcc;">
# TODO: create custom loss function that prioritises objects with a small number of pixels and that are close of other objects
# </div>
#

# %%
dice_loss = sm.losses.DiceLoss(class_weights=weights)  # corrects for class imbalance
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# %% [markdown]
# ### Metrics
# As we are using segmentation for object detection accuracy is not the best metric for model fit. Instead we should use the Intersection over Union (IoU), also known as the Jaccard index. This is the term used to describe the extent of overlap between two boxes (circles in the diagram below). The greater the region of overlap the greater the IoU.
#
# If one box (or circle) is the object, the other is our model prediction. To improve model prediction would be to create two boxes (circles) that completely overlap i.e. the IoU becomes equal to 1, where is varies between 0-1.
#
# This metric is often used with a dice coefficient (which is a loss function - see above).
#
# ![image.png](attachment:57bd2855-dc83-471c-9d5e-e7437f6ea7fa.png)!

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


# %% [markdown]
# ## Model <a name="model"></a>

# %% [markdown]
# ### Adjustable parameters
#
# Epochs = this is how many times the model runs through the training data. Running too few epochs will under fit the model, running too many will overfit. Use callbacks to find the optimum number of epochs - but this will change depending on other input parameters!
#
# Callbacks = Performs actions at the beginning or end of an epoch or batch. The ones used here are early stopping to monitor how the model is performing, if fit isn't improving over epochs then the model will stop (to prevent overfitting, although note will run for 4 more epochs to check it was correct - this is the patience parameter and can be adjusted). The parameter used to monitor if the model has reached it's best is validation loss. Also saving data to tensorboard logs.

# %%
num_epochs = 25

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor="val_loss"),
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
]

# %% [markdown]
# ### Model

# %%
model = get_model()

model.compile(optimizer="adam", loss=total_loss, metrics=metrics)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

model.summary()

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

model.save(
    models_dir.joinpath(f"trail_run_{num_epochs}epochs_{img_size}pix_doolow.hdf5")
)

# %%
y_pred = model.predict(X_test)

# predicted_img = np.argmax(y_pred, axis=0)[:, :]

# %% [markdown]
# ## Output visual checking <a name="output"></a>

# %%
# Crop to size of modelling tile
# test_mask = np.argmax(y_test, axis=3)[0, :, :]
# test_mask.shape

# %%
# "normalised_sat_raster" and "training_mask_raster" only work here because single image used
# TODO: Update to be generalised from X_test and y_test files

fig, axes = plt.subplots(figsize=(13, 8))
plt.subplot(231)
plt.title("Training tile")
plt.imshow(normalised_sat_raster[:, :, 0:3])
plt.subplot(232)
plt.title("Labelled mask")
plt.imshow(training_mask_raster[:, :])
plt.subplot(233)
plt.title("Prediction of class: Non-Building")
plt.imshow(
    y_pred[0, :, :, 0],
    cmap=mpl.colors.LinearSegmentedColormap.from_list("", ["white", col_map(0)]),
)
plt.subplot(234)
plt.title(f"Prediction of class: {building_class_list[0]}")
plt.imshow(
    y_pred[0, :, :, 1],
    cmap=mpl.colors.LinearSegmentedColormap.from_list("", ["white", col_map(0.40)]),
)
plt.subplot(235)
plt.title(f"Prediction of class: {building_class_list[1]}")
plt.imshow(
    y_pred[0, :, :, 2],
    cmap=mpl.colors.LinearSegmentedColormap.from_list("", ["white", col_map(0.60)]),
)
plt.subplot(236)
plt.title(f"Prediction of class: {building_class_list[2]}")
plt.imshow(
    y_pred[0, :, :, 3],
    cmap=mpl.colors.LinearSegmentedColormap.from_list("", ["white", col_map(0.9)]),
)
plt.show()

# %%
