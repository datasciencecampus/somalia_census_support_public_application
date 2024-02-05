# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
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
# This notebook assumes the `1_premodelling_notebook` has already been run and all the training data has been converted into `.npy` arrays. It augments the arrays to bulk out the training data for input into the model. Model parameters that can be adjusted have been laid out in individual cells for ease of optimisation. Finally, model outputs are displayed in a tensorboard and outputted as images below.
#
# #### Contents - to update
#
# 1. ##### [Set-up](#setup)
# 1. ##### [Training parameters](#trainingparameters)
# 1. ##### [Weights](#weights)
# 1. ##### [Model parameters](#modelparameters)
# 1. ##### [Model](#model)
# 1. ##### [Outputs](#output)
#

# %% [markdown]
# ### Checking memory usage

# %%
import os
import psutil

# Get the process ID (PID) of the current Jupyter notebook process
current_pid = os.getpid()

# Get the process memory usage
process = psutil.Process(current_pid)
memory_info = process.memory_info()

# Convert memory usage to gigabytes
memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)

# Print the memory usage in gigabytes
print("Memory usage (gb):", memory_usage_gb)

# %% [markdown]
# ## Set-up <a name="setup"></a>

# %%
# %env TF_GPU_ALLOCATOR=cuda_malloc_async

# %% [markdown]
# ### Import libraries & custom functions

# %%
import random
from pathlib import Path
import datetime

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pynvml
from numba import cuda
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

# %%
from functions_library import get_folder_paths
from multi_class_unet_model_build import jacard_coef, get_model
from loss_functions import get_sm_loss, get_combined_loss, get_focal_tversky_loss

# %% [markdown]
# #### GPU Availability check

# %%
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# %%
physical_devices = tf.config.list_physical_devices("GPU")
try:
    # Disable first GPU
    tf.config.set_visible_devices(physical_devices[0], "GPU")
    logical_devices = tf.config.list_logical_devices("GPU")
except OSError:
    raise RuntimeError(
        "Invalid device or cannot modify virtual devices once initialized."
    )
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


# %% [markdown]
# #### GPU memory usage

# %%
def bytes_to_gb(bytes_value):
    return round(bytes_value / (1024**3), 2)


pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(
    0
)  # Assuming the GPU you are interested in is at index 0

info = pynvml.nvmlDeviceGetMemoryInfo(handle)
total_gb = bytes_to_gb(info.total)
free_gb = bytes_to_gb(info.free)
used_gb = bytes_to_gb(info.used)

print(f"Total GPU Memory: {total_gb} GB")
print(f"Free GPU Memory: {free_gb} GB")
print(f"Used GPU Memory: {used_gb} GB")

pynvml.nvmlShutdown()

# %% [markdown]
# #### Freeing GPU memory

# %%
# reset tensorflow session
tf.keras.backend.clear_session()

# %%
# using CUDA operations
cuda.select_device(0)
cuda.close()

# %% [markdown]
# ### Set-up filepaths

# %%
# for importing stacked arrays
folder_dict = get_folder_paths()
stacked_img = Path(folder_dict["stacked_img_dir"])
stacked_mask = Path(folder_dict["stacked_mask_dir"])

# set model and output directories
models_dir = Path(folder_dict["models_dir"])
outputs_dir = Path(folder_dict["outputs_dir"])

# %% [markdown]
# ### Import arrays

# %% [markdown]
# #### Masks

# %%
ramp = False
if ramp:
    ramp_masks = np.load(stacked_mask / "ramp_bentiu_south_sudan_stacked_masks.npy")
else:
    ramp_masks = None

# %%
training_masks = np.load(stacked_mask / "training_data_all_stacked_masks.npy")
training_masks.shape

# %%
validation_masks = np.load(stacked_mask / "validation_data_all_stacked_masks.npy")
validation_masks.shape

# %%
# joining ramp and training together
# stacked_masks = training_masks
stacked_masks = np.concatenate([training_masks, validation_masks], axis=0)
stacked_masks.shape

# %%
# clearing out some memory
ramp_masks = []
training_masks = []
validation_masks = []

# %% [markdown]
# #### Number of classes

# %%
# number of classes (i.e. building, tent, background)
n_classes = len(np.unique(stacked_masks))
n_classes

# %% [markdown]
# #### Encoding masks

# %%
# encode building classes into training mask arrays
stacked_masks_cat = to_categorical(stacked_masks, num_classes=n_classes)
stacked_masks_cat.shape

# %%
slice_mask = stacked_masks_cat[52]
plt.imshow(slice_mask.argmax(axis=-1))
plt.show()

# %% [markdown]
# #### Images

# %%
if ramp:
    ramp_images = np.load(stacked_img / "ramp_bentiu_south_sudan_stacked_images.npy")
else:
    ramp_images = None

# %%
training_images = np.load(stacked_img / "training_data_all_stacked_images.npy")
training_images.shape

# %%
validation_images = np.load(stacked_img / "validation_data_all_stacked_images.npy")
validation_images.shape

# %%
# dropping 4 th channel to join with ramp
# training_images = training_images[:, :, :, :3]

# %%
# joining training and validation together
# stacked_images = training_images
stacked_images = np.concatenate([training_images, validation_images], axis=0)
stacked_images.shape

# %%
# clearing out some memory
ramp_images = []
training_images = []
validation_images = []

# %% [markdown]
# ### Import filenames

# %%
# ramp_filenames = np.load(
#     stacked_img / "ramp_bentiu_south_sudan_all_stacked_filenames.npy"
# )
training_filenames = np.load(stacked_img / "training_data_all_stacked_filenames.npy")
validation_filenames = np.load(
    stacked_img / "validation_data_all_stacked_filenames.npy"
)

# stacked_filenames = training_filenames
stacked_filenames = np.concatenate(
    [
        training_filenames,
        validation_filenames,
    ],
    axis=0,
)

stacked_filenames.shape

# %% [markdown]
# ### Sense checking images and masks correspond

# %%
# Create a random number to check both image and mask
image_number = random.randint(0, len(stacked_images) - 1)

# Get the BGR image data
bgr_image = stacked_images[image_number, :, :, :3]

# Convert BGR to RGB
rgb_image = bgr_image[:, :, ::-1]

# Normalize the data to the [0.0, 1.0] range
min_value = rgb_image.min()
max_value = rgb_image.max()
image_to_display_normalized = (rgb_image - min_value) / (max_value - min_value)

# Plot the RGB image and mask
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_to_display_normalized)
plt.subplot(122)
plt.imshow(stacked_masks[image_number])
plt.show()


# %% [markdown]
# ## Training parameters <a name="trainingparameters"></a>

# %%
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    stacked_images,
    stacked_masks_cat,
    stacked_filenames,
    test_size=0.20,
    random_state=42,
)

# %% [markdown]
# ## Weights <a name="weights"></a>

# %%
frequency_weights = compute_class_weight(
    "balanced",
    classes=np.unique(stacked_masks),
    y=np.ravel(stacked_masks, order="C"),
)
print(frequency_weights)

# %%
weights = [0.05, 0.05, 0.1, 0.4, 0.4]
frequency_weights = np.array(weights, dtype=np.float64)
print(frequency_weights)

# %% [markdown]
# ## Model parameters <a name="modelparameters"></a>

# %%
# Extract img_height, img_width, and num_channels
img_height, img_width, num_channels = (
    stacked_images.shape[1],
    stacked_images.shape[2],
    stacked_images.shape[3],
)
print(img_height, img_width, num_channels)

# %%
stacked_images = []
stacked_masks_cat = []
stacked_filenames = []

# %%
# defined under training parameters
model = get_model(n_classes, img_height, img_width, num_channels)

# %% [markdown]
# ### Epochs

# %%
# define number of epochs
num_epochs = 300

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
# choose loss function
loss_function = "segmentation_models"  # specify the loss function you want to use: "combined", "segmentation_models, focal_tversky"

optimizer = "adam"  # specify the optimizer you want to use

metrics = ["accuracy", jacard_coef]  # specific the metrics

# %%
loss_weights = frequency_weights

if loss_function == "segmentation_models":
    loss = get_sm_loss(frequency_weights)

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
# Get the current date and time
current_datetime = datetime.datetime.now()
# Format the date and time as a string
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M")

# %%
runid = f"qa_testing_{formatted_datetime}"
runid

# %%
conditions = f"epochs = {num_epochs}\nbatch_size = {batch_size},\nn_classes = {n_classes},\nstacked_array_num = {stacked_masks.shape[0]},\nloss_function = {loss_function},\nweight = {loss_weights}"
print(conditions)

# %%
conditions_filename = outputs_dir / f"{runid}_conditions.txt"
with open(conditions_filename, "w") as f:
    f.write(conditions)


# %% [markdown]
# ## Data Generators  <a name="datagenerator"></a>

# %%
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y


train_gen = DataGenerator(X_train, y_train, 32)
test_gen = DataGenerator(X_test, y_test, 32)


# %% [markdown]
# ## Model <a name="model"></a>

# %%
model.summary()
history1 = model.fit(
    train_gen,
    batch_size=batch_size,
    verbose=1,
    epochs=num_epochs,
    validation_data=test_gen,
    shuffle=False,
    # callbacks=callbacks,
)

# %%
# model.summary()

# history1 = model.fit(
#     X_train,
#     y_train,
#     batch_size=batch_size,
#     verbose=1,
#     epochs=num_epochs,
#     validation_data=(X_test, y_test),
#     shuffle=False,
#     # callbacks=callbacks,
# )

# %%
# optional
# %load_ext tensorboard
# %tensorboard --logdir logs/

# %% [markdown]
# ### Saving files

# %% [markdown]
# #### Saving model

# %%
# saving model run conditions
model_filename = f"{runid}.hdf5"

# save model output into models_dir
model.save(models_dir.joinpath(model_filename))

# %% [markdown]
# #### Saving history

# %%
history = history1


# %%
# saving epochs
history_filename = outputs_dir / f"{runid}.csv"

history_df = pd.DataFrame(history.history)
history_df.to_csv(history_filename, index=False)

# %% [markdown]
# #### Saving outputs

# %%
# defining y_pred first
with tf.device("/cpu:0"):
    y_pred = model.predict(X_test)

# %%
X_test_filename = f"{runid}_xtest.npy"
y_pred_filename = f"{runid}_ypred.npy"
y_test_filename = f"{runid}_ytest.npy"
filenames_test_filename = f"{runid}_filenamestest.npy"

np.save(outputs_dir.joinpath(X_test_filename), X_test)
np.save(outputs_dir.joinpath(y_pred_filename), y_pred)
np.save(outputs_dir.joinpath(y_test_filename), y_test)
np.save(outputs_dir.joinpath(filenames_test_filename), filenames_test)

# %% [markdown]
# ## Clear outputs and remove variables<a name="clear"></a>

# %%
# %reset -f

# %%
