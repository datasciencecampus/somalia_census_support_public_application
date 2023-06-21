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
# 1. ##### [Validation setting](#validation)
# 1. ##### [Data augmentation](#dataaug)
# 1. ##### [Training parameters](#trainingparameters)
# 1. ##### [Weights](#weights)
# 1. ##### [Model parameters](#modelparameters)
# 1. ##### [Model](#model)
# 1. ##### [Outputs](#output)
# 1. ##### [Visual Outputs](#visualoutput)

# %% [markdown]
# ## Set-up <a name="setup"></a>

# %% [markdown]
# ### segmentation_models framework

# %% [markdown]
# ### Import libraries & custom functions

# %%
import random
from pathlib import Path

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.metrics import MeanIoU
from keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# %%
from data_augmentation_functions import (
    stack_array,
    stack_array_with_validation,
    stack_background_arrays,
    hue_shift,
    adjust_brightness,
    adjust_contrast,
    stack_images,
)
from weight_functions import (
    calculate_distance_weights,
    calculate_size_weights,
    calculate_building_distance_weights,
)
from multi_class_unet_model_build import jacard_coef, multi_unet_model, split_data
from loss_functions import get_sm_loss, get_combined_loss, get_focal_tversky_loss

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

# set model and output directories
models_dir = Path.cwd().parent.joinpath("models")
outputs_dir = Path.cwd().parent.joinpath("outputs")

# %% [markdown]
# ## Set validation area <a name="validation"></a>
#
# Option to set specific tiles as validation for model. Currently works by using an area name (i.e. Baidoa). Should be set to 'none' if want to randomly generate training/validation split.

# %%
validation_area = None

# %%
# stacking validation images based on an area
if validation_area is not None:
    validation_images = stack_array_with_validation(img_dir, validation_area)
    validation_images.shape

# %%
# stacking validation masks based on an area
if validation_area is not None:
    validation_masks = stack_array_with_validation(mask_dir, validation_area)
    validation_masks.shape

# %% [markdown]
# ## Data augmentation <a name="dataaug"></a>

# %% [markdown]
# ### Image augmentation

# %%
# creating stack of img arrays that are rotated and horizontally flipped
stacked_images = stack_array(img_dir, validation_area)
stacked_images.shape

# %% [markdown]
# #### Set augmentation

# %%
include_hue_adjustment = True
include_brightness_adjustments = True
include_contrast_adjustments = True
include_backgrounds = True

# %% [markdown]
# #### Hue shift

# %%
if include_hue_adjustment:
    # shift value (between 0 and 1)
    hue_shift_value = 0.5
    adjusted_hue = hue_shift(stacked_images, hue_shift_value)

# %% [markdown]
# #### Brightness

# %%
if include_brightness_adjustments:
    # values <1 will decrease brightness while values >1 will increase brightness
    brightness_factor = 1.5
    adjusted_brightness = adjust_brightness(stacked_images, brightness_factor)

# %% [markdown]
# #### Contrast

# %%
if include_contrast_adjustments:
    # values <1 will decrease contrast while values >1 will increase contrast
    contrast_factor = 2
    adjusted_contrast = adjust_contrast(stacked_images, contrast_factor)

# %% [markdown]
# #### Background images

# %%
if include_backgrounds:
    # creating stack of background img arrays with no augmentation
    background_images = stack_background_arrays(img_dir)

# %% [markdown]
# #### Sense checking hue/brightness/contrast

# %%
# for sense checking brightness/contrast values
random_indices = np.random.choice(len(adjusted_contrast), size=5, replace=False)

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 15))

for i, idx in enumerate(random_indices):
    axes[i, 0].imshow(stacked_images[idx][:, :, :3])
    axes[i, 0].set_title("original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(adjusted_contrast[idx][:, :, :3])
    axes[i, 1].set_title("stacked")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Final image array

# %%
# building images array

all_stacked_images = stack_images(
    stacked_images,
    background_images,
    adjusted_hue,
    adjusted_brightness,
    adjusted_contrast,
    include_hue_adjustment,
    include_backgrounds,
    include_brightness_adjustments,
    include_contrast_adjustments,
)

all_stacked_images.shape

# %% [markdown]
# ### Mask augmentation

# %%
# creating stack of mask arrays that are rotated and horizontally flipped
stacked_masks = stack_array(mask_dir, validation_area)
stacked_masks.shape

# %% [markdown]
# #### Additional augmentations

# %%
# if any of the above image augmentations have been performed then you need corresponding masks
mask_hue, mask_brightness, mask_contrast = [np.copy(stacked_masks) for _ in range(3)]

# %% [markdown]
# #### Background masks

# %%
# creating stack of background img arrays with no augmentation
background_masks = stack_background_arrays(mask_dir)

# %% [markdown]
# #### Final mask array

# %%
# options include:
# background_masks
# mask_hue
# mask_brightness
# mask_contrast

all_stacked_masks = stack_images(
    stacked_masks,
    background_masks,
    mask_hue,
    mask_brightness,
    mask_contrast,
    include_hue_adjustment,
    include_backgrounds,
    include_brightness_adjustments,
    include_contrast_adjustments,
)

all_stacked_masks.shape

# %% [markdown]
# ### Number of classes

# %%
# number of classes (i.e. building, tent, background)
n_classes = len(np.unique(all_stacked_masks))

n_classes

# %% [markdown]
# ### Encoding masks

# %%
# encode building classes into training mask arrays
stacked_masks_cat = to_categorical(all_stacked_masks, num_classes=n_classes)
stacked_masks_cat.shape

# %%
if validation_area is not None:
    # encode building classes into validation masks
    validation_masks_cat = to_categorical(validation_masks, num_classes=n_classes)
    validation_masks_cat.shape

# %% [markdown]
# ### Sense checking images and masks correspond

# %%
# create random number to check both image and mask
image_number = random.randint(0, len(all_stacked_images) - 1)

# plot image and mask
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(all_stacked_images[image_number, :, :, :3])
plt.subplot(122)
plt.imshow(stacked_masks_cat[image_number])
plt.show()

# %% [markdown]
# ## Training parameters <a name="trainingparameters"></a>

# %%
if "validation_images" not in locals() or validation_images is None:
    validation_images = None

if "validation_masks_cat" not in locals() or validation_masks_cat is None:
    validation_masks_cat = None

# %%
X_train, X_test, y_train, y_test = split_data(
    all_stacked_images, stacked_masks_cat, validation_images, validation_masks_cat
)

# %%
img_height, img_width, num_channels = (384, 384, 4)

# %% [markdown]
# ## Weights <a name="weights"></a>

# %% [markdown]
# ### Distance based weighting - adapted from Google

# %%
distance_weights = calculate_distance_weights(stacked_masks_cat, sigma=3, c=100)
print(distance_weights)

# %% [markdown]
# ### Frequency based weighting

# %%
frequency_weights = compute_class_weight(
    "balanced",
    classes=np.unique(all_stacked_masks),
    y=np.ravel(all_stacked_masks, order="C"),
)
print(frequency_weights)

# %% [markdown]
# ### Size based weighting

# %%
size_weights = calculate_size_weights(stacked_masks_cat, alpha=1.0)
print(size_weights)

# %% [markdown]
# ### Building distance weighting

# %%
building_weights = calculate_building_distance_weights(
    stacked_masks_cat, sigma=3, c=200, alpha=1.0
)
print(building_weights)


# %% [markdown]
# ## Model parameters <a name="modelparameters"></a>

# %%
def get_model():
    return multi_unet_model(
        n_classes=n_classes,
        IMG_HEIGHT=img_height,
        IMG_WIDTH=img_width,
        IMG_CHANNELS=num_channels,
    )


# %% [markdown]
# ### Epochs

# %%
# define number of epochs
num_epochs = 200

# %% [markdown]
# ### Batch size

# %%
# define batch size
batch_size = 50

# %% [markdown]
# ### Callbacks

# %%
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor="val_loss"),
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
]

# %% [markdown]
# ### Loss functions

# %%
# defined under training parameters
model = get_model()

# %%
# choose loss function
loss_function = "combined"  # specify the loss function you want to use: "combined", "segmentation_models, focal_tversky"

optimizer = "adam"  # specific the optimizer you want to use

metrics = ["accuracy", jacard_coef]  # specific the metrics

# %%
loss_weights = None

if loss_function == "segmentation_models":
    loss = get_sm_loss(size_weights)

elif loss_function == "combined":
    loss = get_combined_loss()
    loss_weights = [0.5, 0.5]

elif loss_function == "focal_tversky":
    loss = get_focal_tversky_loss()

# %%
model.compile(
    optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics
)

# %% [markdown]
# ### Saving model parameters
#

# %%
runid = "phase_1_w_1_np_21_06_23"

# %%
conditions = f"epochs = {num_epochs}\nbatch_size = {batch_size},\nn_classes = {n_classes},\nstacked_img_num = {all_stacked_masks.shape[0]},\nhue_shift = {hue_shift_value},\nbrightness = {brightness_factor},\ncontrast = {contrast_factor},\nloss_function = {loss_function}"
print(conditions)

# %%
conditions_filename = outputs_dir / f"{runid}_conditions.txt"
with open(conditions_filename, "w") as f:
    f.write(conditions)

# %% [markdown]
# ## Model <a name="model"></a>

# %%
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
# optional
# %load_ext tensorboard
# %tensorboard --logdir logs/

# %% [markdown]
# ### Saving output

# %%
history = history1


# %%
# saving model run conditions
model_filename = f"{runid}.hdf5"

# save model output into models_dir
model.save(models_dir.joinpath(model_filename))

# %%
# saving epochs
history_filename = outputs_dir / f"{runid}.csv"

history_df = pd.DataFrame(history.history)
history_df.to_csv(history_filename, index=False)

# %%
# saving output arrays
# defining y_pred first
y_pred = model.predict(X_test)

X_test_filename = f"{runid}_xtest.npy"
y_pred_filename = f"{runid}_ypred.npy"
y_test_filename = f"{runid}_ytest.npy"

np.save(outputs_dir.joinpath(X_test_filename), X_test)
np.save(outputs_dir.joinpath(y_pred_filename), y_pred)
np.save(outputs_dir.joinpath(y_test_filename), y_test)

# %% [markdown]
# ## Outputs <a name="output"></a>

# %% [markdown]
# ### Training and validation changes

# %%
# create plot showing training and validation loss
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "y", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
# create plot showing the IoU over time
acc = history.history["jacard_coef"]
val_acc = history.history["val_jacard_coef"]

plt.plot(epochs, acc, "y", label="Training IoU")
plt.plot(epochs, val_acc, "r", label="Validation IoU")
plt.title("Training and validation IoU")
plt.xlabel("Epochs")
plt.ylabel("IoU")
plt.legend()
plt.show()

# %% [markdown]
# ### Mean IoU

# %%
# calculating mean IoU

y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)


IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# %% [markdown]
# ### Predications across validation images

# %%
# predict for a few images

# test_img_number = random.randint(0, len(X_test))
test_img_number = 2
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img, 0)
prediction = model.predict(test_img_input)
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

# %% [raw]
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title("Testing Image")
# plt.imshow(test_img[:, :, :3])
# plt.subplot(232)
# plt.title("Testing Label")
# plt.imshow(ground_truth)
# plt.subplot(233)
# plt.title("Prediction on test image")
# plt.imshow(predicted_img)
# plt.show()

# %% [markdown]
# ### Confusion Matrix
#
# More information about [confusion matrices](https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/).

# %%
class_names = ["Background", "Building", "Tent"]

y_true = y_test_argmax
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)

# calculate the confusion matrix
conf_mat = confusion_matrix(y_true.ravel(), y_pred.ravel())

# calculate the precision, recall, and F1-score for each class
num_classes = conf_mat.shape[0]
precision = np.zeros(num_classes)
recall = np.zeros(num_classes)
f1_score = np.zeros(num_classes)

for i in range(num_classes):
    true_positives = conf_mat[i, i]
    false_positives = np.sum(conf_mat[:, i]) - true_positives
    false_negatives = np.sum(conf_mat[i, :]) - true_positives

    precision[i] = true_positives / (true_positives + false_positives)
    recall[i] = true_positives / (true_positives + false_negatives)
    f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

# calculate the accuracy for each class
accuracy = np.zeros(num_classes)
for i in range(num_classes):
    accuracy[i] = conf_mat[i, i] / np.sum(conf_mat[i, :])

# print the results
for i in range(num_classes):
    print(
        f"{class_names[i]} - Precision: {precision[i]}, Recall: {recall[i]}, F1-score: {f1_score[i]}, Accuracy: {accuracy[i]}"
    )


# %%
labels = ["background", "building", "tent"]

# calculate the percentages
row_sums = conf_mat.sum(axis=1)
conf_mat_percent = conf_mat / row_sums[:, np.newaxis]

display = ConfusionMatrixDisplay(
    confusion_matrix=conf_mat_percent, display_labels=labels
)

# plot the confusion matrix
display.plot(cmap="cividis", values_format=".2%")

# show the plot
plt.show()

# %%
