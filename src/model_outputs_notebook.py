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
# ## Set-up

# %% [markdown]
# ### segmentation models framework

# %%
# %env SM_FRAMEWORK = tf.keras

# %% [markdown]
# ### Import libraries & functions

# %%
# Packages
import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from keras.metrics import MeanIoU
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from functions_library import get_folder_paths
from loss_functions import get_combined_loss
from multi_class_unet_model_build import jacard_coef
from model_outputs_functions import compute_class_counts


# %% [markdown]
# ### File directories

# %%
folder_dict = get_folder_paths()

# Set directories to pull run files from
models_dir = Path(folder_dict["models_dir"])
outputs_dir = Path(folder_dict["outputs_dir"])
img_dir = Path(folder_dict["img_dir"])
mask_dir = Path(folder_dict["mask_dir"])

# %% [markdown]
# ## Import data

# %% [markdown]
# ### Set runid

# %%
# Set runid for outputs
runid = "outputs_alt_test"


# %% [markdown]
# ### Model conditions

# %%
model_filename = f"{runid}.hdf5"
model_phase = h5py.File(models_dir.joinpath(model_filename), "r")

# %%
# check loaded in correctly
model_phase.keys()

# %% [markdown]
# ### History (epochs)

# %%
csv_filename = f"{runid}.csv"
history = pd.read_csv(outputs_dir.joinpath(csv_filename))

# %%
# check csv has been loaded
history.head()

# %% [markdown]
# ### Predictions

# %%
X_test_filename = f"{runid}_xtest.npy"
y_pred_filename = f"{runid}_ypred.npy"
y_test_filename = f"{runid}_ytest.npy"
filenames_test_filename = f"{runid}_filenamestest.npy"

X_test = np.load(outputs_dir.joinpath(X_test_filename))
y_pred = np.load(outputs_dir.joinpath(y_pred_filename))
y_test = np.load(outputs_dir.joinpath(y_test_filename))
filenames_test = np.load(outputs_dir.joinpath(filenames_test_filename))


# %%
# check arrays loaded in
filenames_test[4]

# %% [markdown]
# ### Set loss

# %%
# check total loss has loaded successfully
total_loss = get_combined_loss()

# %% [markdown]
# ## Load model

# %%
model = keras.models.load_model(
    model_phase,
    custom_objects={
        "dice_loss_plus_1focal_loss": total_loss[1],
        "focal_loss": total_loss[0],
        "dice_loss": total_loss[1],
        "jacard_coef": jacard_coef,
    },
)


# %% [markdown]
# ## Training and validation changes

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
# ## Compute class outputs

# %%
class_counts_df = compute_class_counts(y_pred, y_test, filenames_test)
# class_counts_df = class_counts_df.drop(columns="Background")
class_counts_df

# %% [markdown]
# ## Confusion Matrix

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
