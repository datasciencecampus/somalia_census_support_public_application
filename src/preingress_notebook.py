# ---
# jupyter:
#   jupytext:
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
# # Pre GCP Ingress Notebook
#
# > Notebook to be run before any files are transferred to the SharePoint GCP ingress folder

# %% [markdown]
# ## Set-up

# %%
import os
from pathlib import Path

import geopandas as gpd

# %%
from functions_library import setup_sub_dir

# %%
data_dir = Path.cwd().parent.joinpath("data")

training_data_dir = data_dir.joinpath("training_data")
img_dir = setup_sub_dir(training_data_dir, "img")
mask_dir = setup_sub_dir(training_data_dir, "mask")

# %% [markdown]
# ## Image file cleaning
#
# * change all file names to lower case
# * check there is a corresponding mask file
# * check banding?
# * ensure naming convention upheld?

# %%
img_count = len(list(img_dir.glob("*.tif")))
mask_count = len(list(mask_dir.glob("*.tif")))

print("there are", img_count, "image files and", mask_count, "mask files")

# %% [markdown]
# ## Mask file cleaning
#
# * change all files to lower case
# * check there is a corresponding image file
# * check there is a type column
# * remove fid or id column
# * check for na
# * ensure naming convention upheld

# %%
for path, subdirs, files in os.walk(training_data_dir):
    dirname = path.split(os.path.sep)[-1]
    if dirname == "mask":
        masks = os.listdir(path)

        for i, mask_name in enumerate(masks):
            if mask_name.endswith(".shp"):

                mask_filename = Path(mask_name).stem

                training_data = gpd.read_file(mask_dir.joinpath(mask_name))

                if "fid" in training_data:
                    training_data = training_data.drop(columns=["fid"])
                else:
                    training_data = training_data.drop(columns=["id"])

                if training_data["Type"].isnull().values.any():
                    print("Oh no")

# %%
