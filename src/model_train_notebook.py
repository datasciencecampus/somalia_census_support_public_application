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
# 1. ##### [Model parameters](#modelparameter)
# 1. ##### [Data augmentation](#dataaug)
# 1. ##### [Loss function](#lossfunction)
# 1. ##### [Metrics](#metrics)
# 1. ##### [Model](#model)
# 1. ##### [Outputs](#output)
# 1. ##### [Visual Outputs](#visualoutput)

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
import random
from pathlib import Path

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models as sm
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
from functions_library import setup_sub_dir
from multi_class_unet_model_build import jacard_coef, multi_unet_model, split_data

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


# set-up model directory for model and outputs
models_dir = setup_sub_dir(Path.cwd().parent, "models")
outputs_dir = setup_sub_dir(Path.cwd().parent, "outputs")

# %% [markdown]
# ## Set model parameters <a name="modelparameter"></a>

# %%
include_hue_adjustment = True
include_backgrounds = True
include_brightness_adjustments = True
include_contrast_adjustments = True

# shift value (between 0 and 1)
hue_shift_value = 0.2

# values <1 will decrease contrast while values >1 will increase contrast
contrast_factor = 2

# values <1 will decrease brightness while values >1 will increase brightness
brightness_factor = 1.5

batch_size = 50

num_epochs = 5

# take the run ID from the excel spreadsheet
runid = ""

# %% [markdown]
# ## Set validation area
#
# An area of Somalia can we set as validation tiles by excluding the area name, as defined below.

# %%
validation_area = None

# %%
# stacking validation images based on an area
validation_images = stack_array_with_validation(img_dir, validation_area)
validation_images.shape

# %%
# stacking validation masks based on an area
validation_masks = stack_array_with_validation(mask_dir, validation_area)
validation_masks.shape

# %% [markdown]
# ## Data augmentation <a name="dataaug"></a>

# %% [markdown]
# U-Net architecture uses max pooling to downsample images across 4 levels, so we need to work with tiles that are divisable by 4 x 2. Training data tiles created in QGIS as ~200m x 200m (which equates to ~400 x 400 pixels as resolution is 0.5m/px). During pre-processing all tiles were resized to 384 x 384.
#
# To input more training data (than we actually have) into the model we augment the existing tiles by rotating and mirroring them. This is done to the `.npy` arrays and then all the arrays are stacked together.

# %% [markdown]
# ### Image augmentation

# %%
# creating stack of img arrays that are rotated and horizontally flipped
stacked_images, stacked_filenames  = stack_array(img_dir, validation_area, expanded_outputs=True)
stacked_images.shape

# %% [markdown]
# #### Additional augmentations

# %%
# hue shifting
adjusted_hue = hue_shift(stacked_images, hue_shift_value)
adjusted_hue.shape

# %%
# adjust brightness
adjusted_brightness = adjust_brightness(stacked_images, brightness_factor)
adjusted_brightness.shape

# %%
# adjust contrast
adjusted_contrast = adjust_contrast(stacked_images, contrast_factor)
adjusted_contrast.shape

# %% [markdown]
# #### Background images

# %%
# creating stack of background img arrays with no augmentation
background_images, background_filenames = stack_background_arrays(img_dir, expanded_outputs=True)

print(len(background_images))

# %% [markdown]
# #### Expand Filenames List

# %%
# Order of Final image array needs to be followed
all_stacked_filenames = np.concatenate([stacked_filenames] + [background_filenames], axis=0)
# Repeats the stacked_filenames to match the number of augmentations
stacked_filenames = np.tile(stacked_filenames, 3)
# Concatenates the remainder
all_stacked_filenames = np.concatenate([all_stacked_filenames] + [stacked_filenames], axis=0)
all_stacked_filenames.shape

# %% [markdown]
# #### Sense checking hue/brightness/contrast

# %%
# for sense checking brightness/contrast values
random_indices = np.random.choice(len(adjusted_hue), size=5, replace=False)

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 15))

for i, idx in enumerate(random_indices):
    axes[i, 0].imshow(stacked_images[idx][:, :, :3])
    axes[i, 0].set_title("original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(adjusted_hue[idx][:, :, :3])
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
# note the below lines do the same thing but has been written out to save time

mask_hue = np.copy(stacked_masks)
mask_brightness = np.copy(stacked_masks)
mask_contrast = np.copy(stacked_masks)

# %% [markdown]
# #### Background masks

# %%
# creating stack of background img arrays with no augmentation
background_masks = stack_background_arrays(mask_dir)

print(len(background_masks))

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
# #### Number of classes

# %%
# number of classes (i.e. building, tent, background)
n_classes = len(np.unique(all_stacked_masks))

n_classes

# %%
# encode building classes into training mask arrays
stacked_masks_cat = to_categorical(all_stacked_masks, num_classes=n_classes)

stacked_masks_cat.shape

# %%
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
X_train, X_test, y_train, y_test, filenames_train, filenames_test = split_data(all_stacked_images, stacked_masks_cat, all_stacked_filenames)

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
    classes=np.unique(all_stacked_masks),
    y=np.ravel(all_stacked_masks, order="C"),
)

# Alternatively, could try balanced weights between classes:
# weights = [0.33, 0.33, 0.33]


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
# define number of epochs, if not already defined above
if "num_epochs" not in locals():
    num_epochs = 150

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
# define batch size, if not already defined above
if "batch_size" not in locals():
    batch_size = 50

# %% [markdown]
# #### Callbacks
# Performs actions at the beginning or end of an epoch or batch. The ones used here are early stopping to monitor how the model is performing, if fit isn't improving over epochs then the model will stop (to prevent overfitting, although note will run for 4 more epochs to check it was correct - this is the patience parameter and can be adjusted). The parameter used to monitor if the model has reached it's best is validation loss. Also saving data to tensorboard logs.

# %%
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=4, monitor="loss"),
    tf.keras.callbacks.TensorBoard(log_dir="logs"),
]

# %% [markdown]
# ### Check model parameters
#

# %%
conditions = f"epochs = {num_epochs}\nbatch_size = {batch_size},\nn_classes = {n_classes},\nhue_shift = {hue_shift_value},\nbrightness = {brightness_factor},\ncontrast = {contrast_factor}, \ninclude_hue_adjustment = {include_hue_adjustment}\ninclude_backgrounds = {include_backgrounds}\ninclude_brightness_adjustments = {include_brightness_adjustments}\ninclude_contrast_adjustments = {include_contrast_adjustments}\nstacked_img_num = {all_stacked_masks.shape[0]}"

print(conditions)

conditions_filename = outputs_dir / f"{runid}_conditions.txt"
with open(conditions_filename, "w") as f:
    f.write(conditions)

# %% [markdown]
# ## Model

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

# %% [markdown]
# #### Saving output

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
filenames_test_filename = f"{runid}_filenamestest.npy"

np.save(outputs_dir.joinpath(X_test_filename), X_test)
np.save(outputs_dir.joinpath(y_pred_filename), y_pred)
np.save(outputs_dir.joinpath(y_test_filename), y_test)
np.save(outputs_dir.joinpath(filenames_test_filename), filenames_test)

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

# %%
import keras
model = keras.models.load_model('/home/jupyter/net-zero-somalia-analysis/somalia_unfpa_census_support/models/phase_1_5_np_07_06_23_validate.hdf5', custom_objects={'dice_loss_plus_1focal_loss': total_loss, 'jacard_coef':jacard_coef})

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
test_img_number = 1
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img, 0)
prediction = model.predict(test_img_input)
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

# %%
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title("Testing Image")
plt.imshow(test_img[:, :, :3])
plt.subplot(232)
plt.title("Testing Label")
plt.imshow(ground_truth)
plt.subplot(233)
plt.title("Prediction on test image")
plt.imshow(predicted_img)
plt.show()

# %% [markdown]
# ### Compute class outputs

# %%
import pandas as pd
import json


def compute_class_counts(y_pred, y_actual):
    """
    Compute the counts of each class in each tile for predicted and test arrays.

    Args:
        y_pred (ndarray): Predicted array with shape (batch_size, height, width, num_classes).
        y_test (ndarray): Actual array with shape (batch_size, height, width, num_classes).


    Returns:
        DataFrame: Pandas DataFrame containing the counts of each class in each sample.

    """

    class_counts_pred = []
    class_counts_actual = []

    class_labels = {0: "Background", 1: "Building", 2: "Tent"}
    
    # Load in the actual feature numbers from geoJSONs
    features_file = mask_dir.joinpath("feature_dict.json")
    with open(features_file) as f:
        feature_data = json.load(f)
    
    # Counts predicted arrays using connected Connected Components

    for tile_index in range(y_pred.shape[0]):
        tile_counts_pred = {}
        tile_counts_actual = {}

        for class_index, class_label in class_labels.items():
            if class_label == "Background":
                continue

            # Extract the predicted mask for the current class
            class_mask_pred = np.argmax(y_pred[tile_index], axis=-1) == class_index

            # Perform connected component analysis for predicted counts
            num_labels_pred, labeled_mask_pred = cv2.connectedComponents(
                class_mask_pred.astype(np.uint8)
            )

            # Count the number of objects for the current class in the current tile (excluding background label)
            num_objects_pred = num_labels_pred - 1

            tile_counts_pred[class_label] = num_objects_pred

        class_counts_pred.append(tile_counts_pred)
        
    
    for filename in filenames_test:
            if filename in feature_data:
                class_counts_actual.append(feature_data[filename])
            else:
                print(filename)

    # Create a pandas DF
    df = pd.DataFrame(columns=["Tile"] + list(class_labels.values()))
  
    # Populate the DataFrame with the actual counts for each class in each tile

    for tile_index in range(y_pred.shape[0]):
        tile_counts_pred = class_counts_pred[tile_index]
        tile_counts_actual = class_counts_actual[tile_index]
        row_data = {"Tile": tile_index}

        for class_label in class_labels.values():
            if class_label != "Background":
                pred_count = tile_counts_pred.get(class_label, 0)
                actual_count = tile_counts_actual.get(class_label, 0)
                row_data[class_label] = pred_count
                row_data[class_label + "_actual"] = actual_count

        df = df.append(row_data, ignore_index=True)

    # probably want to change this when file name added
    df["Tile"] = filenames_test
    df = df.reindex(columns=['Tile', 'Tent', 'Tent_actual', 'Building', 'Building_actual'])
    return df

# %%
class_counts_df = compute_class_counts(y_pred, y_test)
# class_counts_df = class_counts_df.drop(columns="Background")
class_counts_df

# %%
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title("Testing Image")
plt.imshow(X_test[100][:, :, :3])
plt.subplot(232)
plt.title("Testing Label")
plt.imshow(y_test[100])
plt.subplot(233)
plt.title("Prediction on test image")
plt.imshow(y_pred[100])
plt.show()

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
    confusion_matrix=conf_mat_percent,display_labels=labels
)

# plot the confusion matrix
display.plot(cmap="cividis", values_format=".2%")

# show the plot
plt.show()

# %%
