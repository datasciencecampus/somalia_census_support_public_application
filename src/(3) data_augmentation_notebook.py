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
# # (3) Data augmentation
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
# This notebook assumes the `(2) premodelling_notebook` has already been run and all the training data has been converted into `.npy` arrays.
#
# ### Contents
# 1. ##### [Set-up](#setup)
# 1. ##### [Data augmentation](#imageaug)

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
from functions_library import get_data_paths

from data_augmentation_functions import (
    stack_array,
    hue_shift,
    adjust_brightness,
    adjust_contrast,
)


# %% [markdown]
# ### Set-up directories

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
# creating stack of img arrays that are rotated and horizontally flipped
stacked_images, stacked_filenames = stack_array(img_dir, expanded_outputs=True)
stacked_images.shape

# %% [markdown]
# #### Set augmentation

# %%
image_adjustments = {
    "hue_shift": {"enabled": True, "shift_value": 0.5},  # shift value (between 0 and 1)
    "brightness": {
        "enabled": True,
        "factor": 1.5,  # values <1 will decrease brightness while values >1 will increase brightness
    },
    "contrast": {
        "enabled": True,
        "factor": 2,  # values <1 will decrease contrast while values >1 will increase contrast
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
# if image_adjustments["contrast"]["enabled"]:
#     adjusted_contrast = adjust_contrast(stacked_images, image_adjustments["contrast"]["factor"])

# %%
adjusted_contrast = adjust_contrast(
    stacked_images, image_adjustments["contrast"]["factor"]
)

# %% [markdown]
# #### Expand Filenames List

# %%
all_stacked_filenames = []
# Order of Final image array needs to be followed
all_stacked_filenames = np.concatenate(
    [stacked_filenames]
    + [stacked_filenames]
    + [stacked_filenames]
    + [stacked_filenames],
    axis=0,
)

all_stacked_filenames.shape

# %% [markdown]
# #### Final image array

# %%
all_stacked_images = np.concatenate(
    [stacked_images] + [adjusted_hue] + [adjusted_brightness] + [adjusted_contrast],
    axis=0,
)
all_stacked_images.shape

# %% [markdown]
# #### Saving image array

# %%
hue = image_adjustments["hue_shift"]["shift_value"]
brightness = image_adjustments["brightness"]["factor"]
contrast = image_adjustments["contrast"]["factor"]
img_filename = (
    f"all_stacked_images_{folder_dropdown.value}_{hue}_{brightness}_{contrast}.npy"
)

# %%
np.save(img_dir / img_filename, all_stacked_images)

# %% [markdown]
# ### Mask augmentation

# %%
# creating stack of mask arrays that are rotated and horizontally flipped
stacked_masks = stack_array(mask_dir)
stacked_masks.shape

# %% [markdown]
# #### Additional augmentations

# %%
# if any of the above image augmentations have been performed then you need corresponding masks
mask_hue, mask_brightness, mask_contrast = [np.copy(stacked_masks) for _ in range(3)]

# %% [markdown]
# #### Final mask array

# %%
all_stacked_masks = np.concatenate(
    [stacked_masks] + [mask_hue] + [mask_brightness] + [mask_contrast], axis=0
)
all_stacked_masks.shape

# %% [markdown]
# #### Saving mask array

# %%
hue = image_adjustments["hue_shift"]["shift_value"]
brightness = image_adjustments["brightness"]["factor"]
contrast = image_adjustments["contrast"]["factor"]
mask_filename = (
    f"all_stacked_masks_{folder_dropdown.value}_{hue}_{brightness}_{contrast}.npy"
)

# %%
np.save(mask_dir / mask_filename, all_stacked_masks)
