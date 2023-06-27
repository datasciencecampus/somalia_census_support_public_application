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
# # Model outputs
# %% [markdown]
# DESCRIPTION:
#
# TASKS:
# 1. Read in:
#  model (hdf5)
#  history (csv)
#  y_pred, y_test, x_test, filenames_test (4x numpy arrays)
#
#  total_loss
#  jacard_coef
#
# The model is read in with from keras import load model and you'll need to set custom_objects for the jaccard coefficient and loss function. This is actually non-trivial for the loss function (jaccard coefficient a function in the unet .py). I'm going to focus on finishing the custom loss functions, which may fix this but give it a shot and see how far you get.

# %%
# Packages
import os

# %%
os.environ["SM_FRAMEWORK"] = "tf.keras"

# %%
# Packages
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

from pathlib import Path
from functions_library import setup_sub_dir
from loss_functions import dice_loss, focal_loss, get_combined_loss
from multi_class_unet_model_build import jacard_coef
from model_outputs_functions import compute_class_counts

# from keras.models import load_model
from keras.metrics import MeanIoU
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# %% [markdown]
# ##### Setup file directory

# %%
# set data directory
data_dir = Path.cwd().parent.joinpath("data")

# set-up model directory for model and outputs
models_dir = setup_sub_dir(Path.cwd().parent, "models")
outputs_dir = setup_sub_dir(Path.cwd().parent, "outputs")

# set training_data directory within data folder
training_data_dir = data_dir.joinpath("training_data")

# set img and mask directories within training_data directory
img_dir = training_data_dir.joinpath("img")
mask_dir = training_data_dir.joinpath("mask")

# %% [markdown]
# ##### Add runid for outputs

# %%
# Add runid for outputs
runid = "outputs_alt_test"


# %% [markdown]
# ##### Read in parameters and variables

# %%
# for hdf5

model_filename = f"{runid}.hdf5"

model_phase = h5py.File(models_dir.joinpath(model_filename), "r")

# %%
# check loaded in correctly
model_phase.keys()

# %%
# for csv

csv_filename = f"{runid}.csv"

history = pd.read_csv(outputs_dir.joinpath(csv_filename))

# %%
# check csv has been loaded

history.head()

# %%
# for numpy
xtest_npy_filename = "_xtest.npy"
ypred_npy_filename = "_ypred.npy"
ytest_npy_filename = "_ytest.npy"
filenames_test_filename = "_filenamestest.npy"

X_test = np.load(outputs_dir.joinpath(xtest_npy_filename))
y_pred = np.load(outputs_dir.joinpath(ypred_npy_filename))
y_test = np.load(outputs_dir.joinpath(ytest_npy_filename))
filenames_test = np.load(outputs_dir.joinpath(filenames_test_filename))


# %%
# check arrays loaded in
filenames_test[4]

# %% [markdown]
# ### Total Loss

# %%
# check total loss has loaded successfully
total_loss = get_combined_loss()

# %% [markdown]
# #### Load in model with keras

# %%
# hardcoded runind model - when was this changed

model = keras.models.load_model(
    model_phase,
    custom_objects={
        "focal_loss": total_loss[0],
        "dice_loss": total_loss[1],
        "jacard_coef": jacard_coef,
    },
)


# %% [markdown]
# ### Mean IoU

# %%
# calculating mean IoU

n_classes = 3

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
    confusion_matrix=conf_mat_percent, display_labels=labels
)

# plot the confusion matrix
display.plot(cmap="cividis", values_format=".2%")

# show the plot
plt.show()

# %% [markdown]
# ### Training and validation changes

# %%
# create plot showing training and validation loss
loss = history["loss"]
val_loss = history["val_loss"]
epochs = range(1, len(history.loss) + 1)
plt.plot(epochs, history.loss, "y", label="Training loss")
plt.plot(epochs, history.val_loss, "r", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
# create plot showing the IoU over time
acc = history["jacard_coef"]
val_acc = history["val_jacard_coef"]

plt.plot(epochs, history.accuracy, "y", label="Training IoU")
plt.plot(epochs, history.val_accuracy, "r", label="Validation IoU")
plt.title("Training and validation IoU")
plt.xlabel("Epochs")
plt.ylabel("IoU")
plt.legend()
plt.show()

# %% [markdown]
# ### Count Classes

# %%
class_counts_df = compute_class_counts(y_pred, y_test, filenames_test)
# class_counts_df = class_counts_df.drop(columns="Background")
class_counts_df

# %%

# %%
