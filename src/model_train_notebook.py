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

# %%
import random
from datetime import date
from pathlib import Path

# %%
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from keras.metrics import MeanIoU
from keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from data_augmentation_functions import (
    hue_shift_with_alpha,
    stack_array,
    stack_background_arrays,
)

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


# set-up model directory for model and outputs
models_dir = setup_sub_dir(Path.cwd().parent, "models")
outputs_dir = setup_sub_dir(Path.cwd().parent, "outputs")

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
stacked_images = stack_array(img_dir)
stacked_images.shape

# %%
# define the hue shift value
shift_value = 0.2

shifted_images = []
for image in stacked_images:
    shifted_image = hue_shift_with_alpha(image, shift_value)
    shifted_images.append(shifted_image)

# convert the list of shifted images back to a np array
shifted_images = np.array(shifted_images)

# %%
num_images = min(stacked_images.shape[0], 10)
stacked_images = stacked_images[:num_images]
shifted_images = shifted_images[:num_images]


num_cols = min(num_images, 5)
num_rows = int(np.ceil(num_images / num_cols))

fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

for i, ax in enumerate(axs.flatten()):
    if i < num_images:
        ax.imshow(shifted_images[i])
        ax.set_axis_off()
        ax.set_title("Shifted")

    # ax = ax2 = ax.figure.add_subplot(num_rows, num_cols, i + num_images + 1)
    # ax.imshow(stacked_images[i])
    # ax.set_axis_off()
    # ax.set_title('Original')

plt.tight_layout()

plt.show()


# %%
# creating stack of background img arrays with no augmentation
background_images = stack_background_arrays(img_dir)

print(len(background_images))

# %%
# adding augmented arrays and background image arrays together
all_stacked_images = np.concatenate([stacked_images] + [background_images], axis=0)

all_stacked_images.shape

# %% [markdown]
# ### Mask augmentation

# %%
# creating stack of mask arrays that are rotated and horizontally flipped
stacked_masks = stack_array(mask_dir)
stacked_masks.shape

# %%
# creating stack of background img arrays with no augmentation
background_masks = stack_background_arrays(mask_dir)

print(len(background_masks))

# %%
# adding augmented arrays and background image arrays together
all_stacked_masks = np.concatenate([stacked_masks] + [background_masks], axis=0)

all_stacked_masks.shape

# %%
# number of classes (i.e. building, tent, background)
n_classes = len(np.unique(all_stacked_masks))

n_classes

# %%
# encode building classes into training mask arrays
stacked_masks_cat = to_categorical(all_stacked_masks, num_classes=n_classes)

stacked_masks_cat.shape

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
# setting out number of train and validation tiles
X_train, X_test, y_train, y_test = train_test_split(
    all_stacked_images, stacked_masks_cat, test_size=0.20, random_state=42
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
num_epochs = 100

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
# CURRENTLY NO POINT IN SAVING AS WE CAN'T OPEN THE FILES WITH THE DICE LOSS WE USE AND NOT SAVING OUTPUT

# today's date for input into model output
today = date.today().strftime("%Y-%m-%d")
# create filename for model with date and number of epochs
model_filename = f"test_run_{num_epochs}epochs_{today}_all_1.hdf5"

# save model output into models_dir
model.save(models_dir.joinpath(model_filename))

# %% [markdown]
# ## Outputs <a name="output"></a>

# %% [markdown]
# ### Change across epochs

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
