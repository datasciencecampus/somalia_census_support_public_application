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
# # Data augmentation
#
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: #31708f; background-color: #d9edf7; border-color: #bce8f1;">
# Before running this project ensure that the correct kernel is selected (top right). The default project environment name is `venv-somalia-gcp`.
# </div>
#
# **Purpose**
#
# Augments the arrays by rotating, mirroring, and changing brightness/contrast/hue of the arrays
#
# **Things to note**
#
# * This notebook assumes the `1_premodelling_notebook` has already been run and all the training data has been converted into `.npy` arrays.
# * Run final cell to clear variables and outputs
#
# <div class="alert alert-block alert-danger">
#     <i class="fa fa-exclamation-triangle"></i> make sure there are no `stacked_arrays` in your `img_dir
# </div>
#
# <div class="alert alert-block alert-danger">
#     <i class="fa fa-exclamation-triangle"></i> don't run `hue` on `ramp` data as it uses the 4th channel and so won't work
# </div>
#
# ### Contents
# 1. ##### [Set-up](#setup)
# 1. ##### [Data augmentation](#imageaug)
# 1. ##### [Clear outputs & variables](#clear)

# %% [markdown]
# ### Checking memory usage of notebook

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

# %% [markdown]
# ### Import libraries & custom functions

# %%
from pathlib import Path

# %%
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# %%
from functions_library import get_data_paths, get_folder_paths

from data_augmentation_functions import (
    stack_array,
    stack_rotate,
    stack_background_arrays,
    hue_shift,
    adjust_brightness,
    adjust_contrast,
    create_class_borders_array,
)


# %% [markdown]
# ### Set-up directories

# %%
# directories for saving stacked arrays at the end
folder_dict = get_folder_paths()
stacked_img = Path(folder_dict["stacked_img_dir"])
stacked_mask = Path(folder_dict["stacked_mask_dir"])

# %%
# set data directory
data_dir = Path.cwd().parent.joinpath("data")

# get all sub directories within data forlder
sub_dir = [subdir.name for subdir in data_dir.iterdir() if subdir.is_dir()]

# %% [markdown]
# ### Select sub directory

# %%
folder_dropdown = widgets.Dropdown(options=sub_dir, description="select folder:")
display(folder_dropdown)

# %%
# set img and mask directories based on seelcted folder above
img_dir, mask_dir = get_data_paths(folder_dropdown.value)
print(img_dir)
print(mask_dir)

# %% [markdown]
# ## Image augmentation <a name="imageaug"></a>

# %% [markdown]
# #### Rotating & mirroring arrays

# %%
# creating stack of img arrays and filenames
stacked_images, stacked_filenames = stack_array(img_dir, expanded_outputs=True)
stacked_images.shape

# %%
# creating rotaed and mirror array
stacked_rotated, stacked_rotated_filenames = stack_rotate(
    stacked_images, stacked_filenames, expanded_outputs=True
)
stacked_rotated.shape

# %%
# background img arrays
stacked_background, stacked_background_filenames = stack_background_arrays(
    img_dir, expanded_outputs=True
)
stacked_background.shape

# %%
# creating rotated and mirror array
stacked_background_rotated, stacked_background_rotated_filenames = stack_rotate(
    stacked_background, stacked_background_filenames, expanded_outputs=True
)
stacked_background_rotated.shape

# %%
# setting ramp as lower resolution
if folder_dropdown.value == "ramp_bentiu_south_sudan":
    stacked_images = stacked_images.astype(np.float16)
    stacked_rotated = stacked_rotated.astype(np.float16)

# %% [markdown]
# #### Set augmentation

# %%
image_adjustments = {
    "hue_shift": {"enabled": True, "shift_value": 0.2},  # shift value (between 0 and 1)
    "brightness": {
        "enabled": True,
        "factor": 1.5,  # values <1 will decrease brightness while values >1 will increase brightness
    },
}

# %% [markdown]
# #### Hue shift

# %%
adjusted_hue = hue_shift(stacked_images, image_adjustments["hue_shift"]["shift_value"])

# %% [markdown]
# #### Brightness

# %%
adjusted_brightness = adjust_brightness(
    stacked_images, image_adjustments["brightness"]["factor"]
)

# %% [markdown]
# #### Contrast

# %%
adjusted_contrast = adjust_contrast(stacked_images)

# %% [markdown]
# #### Expand Filenames List

# %%
all_stacked_filenames = []
# Order of Final image array needs to be followed
if folder_dropdown.value == "training_data":
    all_stacked_filenames = np.concatenate(
        [stacked_filenames]
        + [stacked_rotated_filenames]
        + [stacked_filenames]
        + [stacked_filenames]
        + [stacked_filenames]
        + [stacked_background_filenames]
        + [stacked_background_rotated_filenames],
        axis=0,
    )

elif folder_dropdown.value == "validation_data":
    all_stacked_filenames = np.concatenate(
        [stacked_filenames]
        + [stacked_rotated_filenames]
        + [stacked_filenames]
        + [stacked_filenames]
        + [stacked_filenames],
        axis=0,
    )

all_stacked_filenames.shape

# %%
file_save = f"{folder_dropdown.value}_all_stacked_filenames.npy"
np.save(stacked_img / file_save, all_stacked_filenames)

# %% [markdown]
# #### Final image array

# %%
if folder_dropdown.value == "training_data":
    all_stacked_images = np.concatenate(
        [stacked_images]
        + [stacked_rotated]
        + [adjusted_hue]
        + [adjusted_brightness]
        + [adjusted_contrast]
        + [stacked_background]
        + [stacked_background_rotated],
        axis=0,
    )

elif folder_dropdown.value == "validation_data":
    all_stacked_images = np.concatenate(
        [stacked_images]
        + [stacked_rotated]
        + [adjusted_hue]
        + [adjusted_brightness]
        + [adjusted_contrast],
        axis=0,
    )

all_stacked_images.shape

# %% [markdown]
# #### Padding

# %%
padding = 20
all_stacked_images = np.pad(
    all_stacked_images,
    ((0, 0), (padding, padding), (padding, padding), (0, 0)),
    mode="constant",
)
all_stacked_images.shape

# %% [markdown]
# #### Saving image array

# %%
img_filename = f"{folder_dropdown.value}_all_stacked_images.npy"

# %%
np.save(stacked_img / img_filename, all_stacked_images)

# %%
# clearing memory
adjusted_hue = []
adjusted_brightness = []
adjusted_contrast = []
stacked_images = []

# %% [markdown]
# ### Mask augmentation

# %%
# creating stack of mask arrays
stacked_masks = stack_array(mask_dir)
stacked_masks.shape

# %%
# creating rotaed and mirror array
mask_rotated = stack_rotate(stacked_masks, stacked_filenames)
mask_rotated.shape

# %%
# background mask arrays
mask_background = stack_background_arrays(mask_dir)
mask_background.shape

# %%
# creating rotated and mirror array
mask_background_rotated = stack_rotate(mask_background, stacked_filenames)
mask_background_rotated.shape

# %% [markdown]
# #### Additional augmentations

# %%
# if any of the above image augmentations have been performed then you need corresponding masks
mask_hue, mask_brightness, mask_contrast = [np.copy(stacked_masks) for _ in range(3)]

# %% [markdown]
# #### Final mask array

# %%
if folder_dropdown.value == "training_data":
    all_stacked_masks = np.concatenate(
        [stacked_masks]
        + [mask_rotated]
        + [mask_hue]
        + [mask_brightness]
        + [mask_contrast]
        + [mask_background]
        + [mask_background_rotated],
        axis=0,
    )

elif folder_dropdown.value == "validation_data":
    all_stacked_masks = np.concatenate(
        [stacked_masks]
        + [mask_rotated]
        + [mask_hue]
        + [mask_brightness]
        + [mask_contrast],
        axis=0,
    )

all_stacked_masks.shape

# %%
# all_stacked_masks = all_stacked_masks.astype(np.int32)
all_stacked_masks.nbytes

# %% [markdown]
# #### Create border classes

# %%
# create border of original polygon then reduce polygon by 2 in all_stacked_masks
all_stacked_masks, all_stacked_edges = create_class_borders_array(all_stacked_masks)

# %%
all_stacked_masks.shape

# %%
all_stacked_edges.shape

# %% [markdown]
# #### Padding

# %%
all_stacked_masks = np.pad(
    all_stacked_masks, ((0, 0), (padding, padding), (padding, padding)), mode="constant"
)
all_stacked_masks.shape

# %%
all_stacked_edges = np.pad(
    all_stacked_edges, ((0, 0), (padding, padding), (padding, padding)), mode="constant"
)
all_stacked_edges.shape

# %% [markdown]
# #### Saving mask array

# %%
mask_filename = f"{folder_dropdown.value}_all_stacked_masks.npy"

# %%
np.save(stacked_mask / mask_filename, all_stacked_masks)

# %% [markdown]
# #### Saving edge array

# %%
edge_filename = f"{folder_dropdown.value}_all_stacked_edges.npy"

# %%
np.save(stacked_mask / edge_filename, all_stacked_edges)

# %% [markdown]
# ## Clear outputs and remove variables<a name="clear"></a>

# %%
# %reset -f


# %%
