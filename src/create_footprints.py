# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: venv-somalia-gcp (Local)
#     language: python
#     name: venv-somalia-gcp
# ---

# %% [markdown]
# ## Notebook for creating footprints
#
# <div class="alert alert-block altert-danger">
#     <i class="fa fa-exclamation-triangle"></i> check the kernel in the above right is`venv-somalia-gcp`
# </div>
#
# **Purpose**
#
# To use a pretrained model to create shelter footprints for an individual Planet image
#
# **Things to note**
#
# - You need to manually select the model you want to use - and ensure it's saved in the `models` directory
# - The Planet image needs to have been broken down into 384 x 384 tiles locally (`create_input_tiles`) and ingressed into local GCP storage

# %% [markdown]
# ## Set-up

# %%
import numpy as np

import keras
from pathlib import Path
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import rasterio as rio
import folium
import shutil


# %%
from functions_library import get_folder_paths
from loss_functions import get_loss_function
from multi_class_unet_model_build import jacard_coef
from data_augmentation_functions import stack_array
from create_footprint_functions import (
    process_tile,
    process_image_files,
    check_shapes,
    filter_empty_arrays,
    extract_transform_from_directory,
    modify_transform,
)

# %%
folder_dict = get_folder_paths()
models_dir = Path(folder_dict["models_dir"])
outputs_dir = Path(folder_dict["outputs_dir"])

camp_tiles_dir = Path(folder_dict["camp_tiles_dir"])
footprints_dir = Path(folder_dict["footprints_dir"])

# %%
# Set model variables
n_classes = 3
runid = "footprint_runs_2024-06-06_1241"
padding = 8

# %% [markdown]
# ## Load model

# %%
# find what loss functions was used
conditions_path = outputs_dir / f"{runid}_conditions.txt"
with open(conditions_path, "r") as file:
    text = file.read()
print(text)

# %%
loss_options = (
    "dice",
    "focal",
    "combined",
    "segmentation_models",
    "custom",
    "tversky",
    "focal_tversky",
)
loss_dropdown = widgets.Dropdown(
    options=loss_options, description="select loss function:"
)
display(loss_dropdown)

# %%
loss = get_loss_function(loss_dropdown.value)


# %%
model_filename = f"{runid}.h5"
model_phase = models_dir / model_filename

# %%
model = keras.models.load_model(
    model_phase,
    custom_objects={
        "focal_loss": loss,
        "dice_loss": loss,
        "jacard_coef": jacard_coef,
    },
)

# %% [markdown]
# ## Select area folder

# %%
# get all sub directories within camp tiles folder
sub_dir = [subdir.name for subdir in camp_tiles_dir.iterdir() if subdir.is_dir()]

folder_dropdown = widgets.Dropdown(options=sub_dir, description="select folder:")
display(folder_dropdown)

# %%
area = folder_dropdown.value
area_dir = camp_tiles_dir / area
print(area_dir)

# %% [markdown]
# ### Move files into sub area directories

# %%
# list of img files
img_files = list(area_dir.glob("*.tif"))

# create folders based on second word & move files
for file in img_files:
    first_two_words = file.stem.split("_")[:2]
    area_subdir = "_".join(first_two_words)

    destination_dir = area_dir / area_subdir
    destination_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(str(file), str(destination_dir / file.name))

# %% [markdown]
# ## Select sub area folder

# %%
# get all sub directories within camp tiles folder
sub_dir = [subdir.name for subdir in area_dir.iterdir() if subdir.is_dir()]

sub_folder_dropdown = widgets.Dropdown(options=sub_dir, description="select folder:")
display(sub_folder_dropdown)

# %%
sub_area = sub_folder_dropdown.value
sub_area_dir = area_dir / sub_area
print(sub_area_dir)

# %% [markdown]
# ### Move the polygon tiles into a separate folder
#
# > want to experiment with just using polygons not tiles hence keeping these for now

# %%
polygon_directory = sub_area_dir / "polygon_tiles"
polygon_directory.mkdir(exist_ok=True)

for file in sub_area_dir.glob("*.tif"):
    if "polygons" in file.stem:
        destination_path = polygon_directory / file.name
        shutil.move(file, destination_path)
        print(f"Moved {file.name} to {polygon_directory}")

# %% [markdown]
# ### Get image crs

# %%
tiff_files = list(sub_area_dir.glob("*.tif"))
if not tiff_files:
    print("No GeoTIFF files found in the directory.")
    exit()

# Read the first GeoTIFF file
first_tiff_file = tiff_files[0]

# Open the GeoTIFF file
with rio.open(first_tiff_file) as src:
    # Get CRS
    crs = src.crs

print(f"CRS for {sub_area}:", crs)

# %% [markdown]
# ## Process img files
#
# > Put `.geotiff` through same process as training images and output as `.npy`

# %%
# list all .tif files in directoy
img_files = list(sub_area_dir.glob("*.tif"))
img_size = 384

# %%
# create npy arrays & highlight any files with issues
error_files = process_image_files(img_files, img_size, sub_area_dir)


# %%
# check all npy files have the same shape
check_shapes(sub_area_dir)

# %% [markdown]
# ### Create stacked arrays and delete `.npy` files in directory

# %%
unseen_images, unseen_filenames = stack_array(sub_area_dir, expanded_outputs=True)
print(unseen_images.shape)
print(unseen_filenames.shape)

# %%
# delete npy arrays now stacked
for file in Path(area_dir).glob("*npy"):
    file.unlink()

# %% [markdown]
# ### Add padding

# %%
padded_unseen_images = np.pad(
    unseen_images,
    ((0, 0), (padding, padding), (padding, padding), (0, 0)),
    mode="constant",
)
padded_unseen_images.shape

# %% [markdown]
# ### Check for any blank arrays

# %%
filtered_images, filtered_filenames = filter_empty_arrays(
    padded_unseen_images, unseen_filenames
)


# %% [markdown]
# ### Clear memory

# %%
unseen_images = []
padded_unseen_images = []
unseen_filenames = []

# %% [markdown]
# ## Convert to polygons

# %%
# get transformation matrix from original .tiff images
transforms = extract_transform_from_directory(sub_area_dir)

# modaify transformation matrices to account for padding
modified_transforms = dict(
    (k, modify_transform(v, padding)) for k, v in transforms.items()
)

# %%
# create georeferenced footpritns
unique_classes = list(range(n_classes))
all_results = []
for idx, (tile, filename) in enumerate(zip(filtered_images, filtered_filenames)):
    result_gdf = process_tile(
        model, tile, unique_classes, filename, idx, modified_transforms, crs
    )
    if result_gdf is not None:
        all_results.append(result_gdf)

all_polygons_gdf = pd.concat(all_results, ignore_index=True)


# %% [markdown]
# ### Save polygons for outputting

# %%
output_footprints = footprints_dir / f"{sub_folder_dropdown.value}_footprints.geojson"
all_polygons_gdf.to_file(output_footprints, driver="GeoJSON")

# %%
output_footprints

# %% [markdown]
# ## Building counts & plotting

# %%
building_count = all_polygons_gdf["type"].value_counts().get("buildings", 0)
tent_count = all_polygons_gdf["type"].value_counts().get("tents", 0)

print("Number of buildings:", building_count)
print("Number of tents:", tent_count)

# %%
filtered_gdf = all_polygons_gdf[all_polygons_gdf["index_num"] == 0]

# change crs into lat/long for plotting
filtered_gdf = filtered_gdf.to_crs(epsg=4326)

# creating multipolygons for plotting testing
dissolved_gdf = filtered_gdf.dissolve(by="type")

# %%

mymap = folium.Map(
    location=[
        filtered_gdf.geometry.centroid.y.mean(),
        filtered_gdf.geometry.centroid.x.mean(),
    ],
    zoom_start=10,
)

for idx, row in filtered_gdf.iterrows():
    folium.GeoJson(row["geometry"]).add_to(mymap)

mymap

# %%
