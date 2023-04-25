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
# While the overall aim is to apply the model across all IDP camps in Somalia, this feasibility study focuses on 5 areas of interest:
# * Baidoa
# * Beledweyne
# * Doolow
# * Kismayo
# * Mogadishu
#
# These areas of interest (plus Doolow) were the subject of a recent Somalia National Bureau of Statistics (SNBS) survey, and so, provide a unique opportunity to add some element of ground-truthed data to the model.
#
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: #31708f; background-color: #d9edf7; border-color: #bce8f1;">
# Before running this project ensure that the correct kernel is selected (top right). The default project environment name is `venv-somalia-gcp`.
# </div>
#
# This notebook assumes the `premodelling_notebook` has already been run and all the training data has been converted into `.npy` arrays. It augments the arrays to bulk out the training data for input into the model. Model parameters that can be adjusted have been laid out in individual cells for ease of optimisation. Finally, model outputs are displayed in a tensorboard and outputted as images below.
#
# ## Contents
#
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Validation file](#validation)
# 1. ##### [Data Augmentation](#dataaug)
# 1. ##### [Loss function](#lossfunction)
# 1. ##### [Metrics](#metrics)
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
from pathlib import Path

# %%
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
from multi_class_unet_model_build import jacard_coef, multi_unet_model

# %% [markdown]
# ### Set-up filepaths

# %%
# set data directory
data_dir = Path.cwd().parent.joinpath("data")

# set training_data directory within data folder
training_data_dir = data_dir.joinpath("training_data")

# set img and mask directories within training_data directory
img_dir = training_data_dir.joinpath("img")
mask_dir = training_data_dir.joinpath("mask")


# set-up model directory for model outputs
models_dir = setup_sub_dir(Path.cwd().parent, "models")

# %% [markdown]
# ## Set validation file <a name="validation"></a>
#
# > this will be removed when there is more data

# %%
# set validation file - until we have more training data
validation_image = img_dir.joinpath("training_data_doolow_1_jo_bgr.npy")
validation_mask = mask_dir.joinpath("training_data_doolow_1_jo_bgr_mask.npy")

# %% [markdown]
# ## Data augmentation <a name="dataaug"></a>

# %% [markdown]
# ### Image augmentation
#
# U-Net architecture uses max pooling to downsample images across 4 levels, so we need to work with tiles that are divisable by 4 x 2. Training data tiles created in QGIS as ~200m x 200m (which equates to ~400 x 400 pixels as resolution is 0.5m/px). During pre-processing all tiles were resized to 384 x 384.
#
# To input more training data (than we actually have) into the model we augment the existing tiles by rotating and mirroring them. This is done to the `.npy` arrays and then all the arrays are stacked together.

# %%
# list all .npy files in img_dir but ignore the image we are using for validation
# TODO: CHANGE train/val split when there is more data

# removing all the background images as need to fix the mask situation
image_files = [
    f
    for f in img_dir.glob("**/*.npy")
    if "background" not in f.parts and f.name != validation_image
]
# when above fixed can use line below
# image_files = [f for f in img_dir.glob("*.npy") if f.name != validation_image]

# empty list to store images
image_arrays = []

for file in image_files:

    # load the array
    img_arr = np.load(file)

    # check array has the correct shape
    if img_arr.shape != (384, 384, 4):
        print(f"skipping {file} due to incorrect shape: {img_arr.shape}")

    # rotate the array by 90 degrees and add to list
    image_arrays.append(np.rot90(img_arr, k=1, axes=(0, 1)))

    # rotate the array by 180 degrees and add to list
    image_arrays.append(np.rot90(img_arr, k=2, axes=(0, 1)))

    # rotate the array by 270 degrees and add to list
    image_arrays.append(np.rot90(img_arr, k=3, axes=(0, 1)))

    # flip the array horizontally and add to list
    image_arrays.append(np.fliplr(img_arr))

    # flip the array verticallyand add to list
    image_arrays.append(np.flipud(img_arr))

# stack all the images in the list
stacked_images = np.stack(image_arrays)

print(stacked_images.shape)

# %%
# reorder the array to image list index, height, width, categorical class
# stacked_training_rasters = np.transpose(stacked_images, axes=[3, 0, 1, 2])

# stacked_training_rasters.shape

# %% [markdown]
# ### Mask augmentation
#
# Performing same augmentation on masks as images.

# %%
# list all .npy files in mask_dir but ignore the image we are using for validation
# TODO: CHANGE train/val split when there is more data
mask_files = [f for f in mask_dir.glob("*.npy") if f.name != validation_mask]

# empty list to store masks
mask_arrays = []

for file in mask_files:

    # load the array
    mask_arr = np.load(file)

    # check array has the correct shape
    if mask_arr.shape != (384, 384):
        print(f"skipping {file} due to incorrect shape: {mask_arr.shape}")
        continue

    # rotate the array by 90 degrees and add to list
    mask_arrays.append(np.rot90(mask_arr, k=1, axes=(0, 1)))

    # rotate the array by 180 degrees and add to list
    mask_arrays.append(np.rot90(mask_arr, k=2, axes=(0, 1)))

    # rotate the array by 270 degrees and add to list
    mask_arrays.append(np.rot90(mask_arr, k=3, axes=(0, 1)))

    # flip the array horizontally and add to list
    mask_arrays.append(np.fliplr(mask_arr))

    # flip the array verticallyand add to list
    mask_arrays.append(np.flipud(mask_arr))

# stack all the mask in the list
stacked_masks = np.stack(mask_arrays)

print(stacked_masks.shape)

# %%
# number of classes (i.e. building, tent, background)
n_classes = len(np.unique(stacked_masks))

# create colour map for preservation at point of displaying classes
# TODO: MOVE to more logical place
col_map = mpl.cm.get_cmap("viridis", n_classes)

n_classes

# %%
# encode building classes into training mask arrays
stacked_masks_cat = to_categorical(stacked_masks, num_classes=n_classes)

# reorder array to list index, height, width, categorical class - used in original but might not be needed anymore?
# masks_categorical = np.transpose(labels_categorical, axes=[3, 0, 1, 2])

stacked_masks_cat.shape

# %% [markdown]
# ## Training parameters <a name="trainingparameters"></a>

# %%
# to use this remove the validation image
X_train, X_test, y_train, y_test = train_test_split(
    stacked_images, stacked_masks_cat, test_size=0.20, random_state=42
)

# %%
img_height, img_width, num_channels = (384, 384, 4)


# %%
def get_model():
    return multi_unet_model(
        n_classes=n_classes,
        IMG_HEIGHT=img_height,
        IMG_WIDTH=img_width,
        IMG_CHANNELS=num_channels,
    )


# %% [markdown]
# ## Loss function <a name="lossfunction"></a>
# There are different types of loss functions that we can use in semantic segmentation either separately or combined together. Here, we combine dice and focal loss.

# %% [markdown]
# ### Dice loss
#
# Weighted dice loss allows for the class frequencies from the training masks to be calculated (i.e. how often does a building type appear) and sets weights based on this.

# %%
# calculating weight for dice loss function
weights = compute_class_weight(
    "balanced",
    classes=np.unique(stacked_masks_cat),
    y=np.ravel(stacked_masks_cat, order="C"),
)

# Alternatively, could try balanced weights between classes:
# weights = [0.25, 0.25, 0.25, 0.25]

# only producing 2 weights?
print(weights)

# %% [markdown]
# ### Focal loss
#
# Here we are using focal loss, an extension of cross-entropy loss. By default, focal loss reduces the weights of well-classified objects, those that have a probability >0.5, and increases the weights of objects with a probability of <0.5. In practice what this means is it reduces the weight (i.e. spends less time focusing on) easily classifable objects (i.e. service buildings) and instead focuses on hard to classify objects (i.e. tents).
#
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: #8a6d3b;; background-color: #fcf8e3; border-color: #faebcc;">
# TODO: create custom loss function that prioritises objects with a small number of pixels and that are close of other objects
# </div>

# %%
dice_loss = sm.losses.DiceLoss(class_weights=weights)  # corrects for class imbalance
focal_loss = sm.losses.CategoricalFocalLoss()
# combined loss function
total_loss = dice_loss + (1 * focal_loss)

# %% [markdown]
# ## Metrics <a name="metrics"></a>
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
# ## Model <a name="model"></a>

# %% [markdown]
# ### Adjustable parameters

# %% [markdown]
# #### Epochs
# This is how many times the model runs through the training data. Running too few epochs will under fit the model, running too many will overfit. Use callbacks to find the optimum number of epochs - but this will change depending on other input parameters!

# %%
num_epochs = 25

# %% [markdown]
# #### Callbacks
# Performs actions at the beginning or end of an epoch or batch. The ones used here are early stopping to monitor how the model is performing, if fit isn't improving over epochs then the model will stop (to prevent overfitting, although note will run for 4 more epochs to check it was correct - this is the patience parameter and can be adjusted). The parameter used to monitor if the model has reached it's best is validation loss. Also saving data to tensorboard logs.

# %%
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor="val_loss"),
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
]

# %% [markdown]
# #### Batch size
#
# The batch size is the number of samples propagated through each epoch. For example, if you have `batch_size = 100` then the model is trained using the first 100 training tiles, then it takes the next 100 samples and trains the model again etc. Using batch sizes smaller than the sample size requires less memory and trains the model quicker in its mini-batches, as weights are updated after each batch.
#
# The larger the batch size the better but the more memory it will use.
#
# [32, 64]- Good starters
#
# [32, 64] - CPU
#
# [128, 256] - GPU territory

# %%
batch_size = 32

# %% [markdown]
# ## Model
#
# > Everything from here down needs some love

# %%
# defined under training parameters
model = get_model()

model.compile(optimizer="adam", loss=total_loss, metrics=metrics)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

model.summary()

history1 = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
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

model.save(models_dir.joinpath(f"trail_run_{num_epochs}epochs_pix_doolow.hdf5"))

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

building_class_list = ["Building", "Tent"]

# fig, axes = plt.subplots(figsize=(13, 8))
# plt.subplot(231)
plt.title("Training tile")
plt.imshow(validation_image[:, :, 0:3])
plt.subplot(232)
plt.title("Labelled mask")
plt.imshow(validation_mask[:, :])
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
