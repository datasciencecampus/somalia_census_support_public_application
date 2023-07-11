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

from functions_library import get_folder_paths
from loss_functions import get_combined_loss
from multi_class_unet_model_build import jacard_coef
from model_outputs_functions import (
    calculate_metrics,
    plot_confusion_matrix,
    remove_rows_by_index,
    compute_predicted_counts,
    compute_actual_counts,
    compute_object_counts,
)


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
runid = "phase_1_gpu_1_28_06_23"


# %% [markdown]
# ### Model conditions

# %%
model_filename = f"{runid}.hdf5"
model_phase = h5py.File(models_dir.joinpath(model_filename), "r")

# %%
# check loaded in correctly
# model_phase.keys()

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
        "dice_loss_plus_1focal_loss": total_loss,
        "dice_loss_plus_focal_loss": total_loss,
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

# argmax
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
# not argmax
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
# ## Confusion Matrix

# %%
class_names = ["Background", "Building", "Tent"]

y_true = y_test_argmax
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)

metrics = calculate_metrics(y_true, y_pred, class_names)
metrics = metrics.set_index("Class")
metrics

# %%
labels = ["background", "building", "tent"]

plot_confusion_matrix(y_true, y_pred, labels, show_percentages=True)


# %% [markdown]
# ## Compute class outputs

# %%
# to remove background tiles
words_to_remove = "background"

# %% [markdown]
# ### Actual from JSON

# %%
df_json = compute_actual_counts(filenames_test)
df_json_filtered = remove_rows_by_index(df_json, words_to_remove)
df_json_filtered = df_json_filtered[~df_json_filtered.index.duplicated()]

# %% [markdown]
# ### Connected components

# %%
df_connected = compute_predicted_counts(y_pred, filenames_test)

df_connected_filtered = remove_rows_by_index(df_connected, words_to_remove)
df_connected_final = df_connected_filtered.join(df_json_filtered)
df_connected_final

# %% [markdown]
# ### Pixel counts

# %%
average_building_size = 100
average_tent_size = 6

df_pixel = compute_object_counts(
    y_pred, filenames_test, average_building_size, average_tent_size
)
df_pixel_filtered = remove_rows_by_index(df_pixel, words_to_remove)
df_pixel_final = df_pixel_filtered.join(df_json_filtered)
df_pixel_final
