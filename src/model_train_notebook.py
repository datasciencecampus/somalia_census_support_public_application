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

import random
from datetime import date

# %%
from pathlib import Path

# %%
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from keras.metrics import MeanIoU
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
# ## Data augmentation <a name="dataaug"></a>

# %% [markdown]
# ### Image augmentation
#
# U-Net architecture uses max pooling to downsample images across 4 levels, so we need to work with tiles that are divisable by 4 x 2. Training data tiles created in QGIS as ~200m x 200m (which equates to ~400 x 400 pixels as resolution is 0.5m/px). During pre-processing all tiles were resized to 384 x 384.
#
# To input more training data (than we actually have) into the model we augment the existing tiles by rotating and mirroring them. This is done to the `.npy` arrays and then all the arrays are stacked together.

# %%
# all .npy files in a directory
# image_files = list(img_dir.glob('*.npy'))

# read in all .npy files except those that are just background
image_files = [
    file for file in img_dir.glob("*.npy") if not file.name.endswith("background.npy")
]

# sort the file names alphabetically
image_files = sorted(image_files)

# empty list for appending original images
image_arrays = []

# load each .npy and append to a list
for file in image_files:
    np_array = np.load(file)
    image_arrays.append(np_array)


# create a rotated version of each image and stack along the same axis
rotations = []
for i in range(4):
    rotated = np.rot90(image_arrays, k=1, axes=(1, 2))
    if i > 0:
        rotated = np.fliplr(rotated)
    rotations.append(rotated)

# create horizontal mirror of each image and stack along the same axis
mirrors = [
    np.fliplr(image_arrays),
    np.fliplr(rotations[0]),
    np.fliplr(rotations[1]),
    np.fliplr(rotations[2]),
]

# stack the original arrays, rotated versions and mirror versions
stacked_images = np.concatenate([image_arrays] + rotations + mirrors, axis=0)

stacked_images.shape

# %%
# create a 2 x 5 grid
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

# loop over the first 10 images in the array and plot each
for i in range(10):

    # compute the row and column index in the grid
    row = i // 5
    col = i % 5

    # select the i-th image from the array
    img = stacked_images[i, :, :, :3]

    # normalise the image data to the range of 0 to 1
    img_normalised = img.astype(np.float32) / np.max(img)

    # plot the image
    axs[row, col].imshow(img_normalised)

# show the plot
plt.show()

# %% [markdown]
# ### Mask augmentation
#
# Performing same augmentation on masks as images.

# %%
# all .npy files in a directory
# mask_files = list(mask_dir.glob('*.npy'))

# read in all .npy files except those that are just background
mask_files = [
    file
    for file in mask_dir.glob("*.npy")
    if not file.name.endswith("background_mask.npy")
]

# sort the file names alphabetically
mask_files = sorted(mask_files)

# empty list for appending original images
mask_arrays = []

# load each .npy and append to a list
for file in mask_files:
    np_array = np.load(file)
    mask_arrays.append(np_array)

# create a rotated version of each image and stack along the same axis
rotations = []
for i in range(4):
    rotated = np.rot90(mask_arrays, k=1, axes=(1, 2))
    if i > 0:
        rotated = np.fliplr(rotated)
    rotations.append(rotated)

# create horizontal mirror of each image and stack along the same axis
mirrors = [
    np.fliplr(mask_arrays),
    np.fliplr(rotations[0]),
    np.fliplr(rotations[1]),
    np.fliplr(rotations[2]),
]

# stack the original arrays, rotated versions and mirror versions

stacked_masks = np.concatenate([mask_arrays] + rotations + mirrors, axis=0)

stacked_masks.shape

# %%
# number of classes (i.e. building, tent, background)
n_classes = len(np.unique(stacked_masks))

n_classes

# %%
# encode building classes into training mask arrays
stacked_masks_cat = to_categorical(stacked_masks, num_classes=n_classes)

stacked_masks_cat.shape

# %%
# create random number to check both image and mask
image_number = random.randint(0, len(stacked_images))

# plot image and mask
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(stacked_images[image_number, :, :, :3])
plt.subplot(122)
plt.imshow(stacked_masks_cat[image_number])
plt.show()

# %% [markdown]
# ## Training parameters <a name="trainingparameters"></a>

# %%
# setting out number of train and validation tiles
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
# calculating weight for dice loss function - only producing 2 weights
weights = compute_class_weight(
    "balanced",
    classes=np.unique(stacked_masks_cat),
    y=np.ravel(stacked_masks_cat, order="C"),
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
num_epochs = 50

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
batch_size = 45

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
# today's date for input into model output
today = date.today().strftime("%Y-%m-%d")
# create filename for model with date and number of epochs
model_filename = f"test_run_{num_epochs}epochs_{today}.hdf5"

# save model output into models_dir
model.save(models_dir.joinpath(model_filename))

# %%
y_pred = model.predict(X_test)

# predicted_img = np.argmax(y_pred, axis=0)[:, :]

# %% [markdown]
# ## Outputs <a name="output"></a>

# %%
# create plot showing training and validation loss
history = history1
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
# calculating mean IoU

y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)


IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# %%
# predict for a few images

test_img_number = random.randint(0, len(X_test))
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
plt.imshow(test_img)
plt.subplot(232)
plt.title("Testing Label")
plt.imshow(ground_truth)
plt.subplot(233)
plt.title("Prediction on test image")
plt.imshow(predicted_img)
plt.show()

# %% [markdown]
# ## Visual outputs <a name="visualoutput"></a>

# %%
# rescale the pixel values to [0,1]
output = y_pred / np.max(y_pred)

# define a list of labels to assign to each subplot - don't currently know what each is
labels = [
    "Image 1",
    "Image 2",
    "Image 3",
    "Image 4",
    "Image 5",
    "Image 6",
    "Image 7",
    "Image 8",
    "Image 9",
]

# create a figure with 3 rows and 3 columns to display the 9 output images
fig, ax = plt.subplots(3, 3, figsize=(10, 10))

# iterate over the rows and columns of the subplot array
for i in range(3):
    for j in range(3):

        # select the ith and jth output image
        ax[i, j].imshow(output[i * 3 + j])
        # ax[i, j].axis('off')

        # add a title to the subplot
        ax[i, j].set_title(labels[i * 3 + j])

# add a main title to the figure
fig.suptitle("U-Net Model Output")

# adjust the spacing between subplots
fig.tight_layout()

# show the plot
plt.show()

# %%
