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
from keras.models import load_model
from pathlib import Path
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import rasterio as rio
import folium
import shutil


# %%
from functions_library import get_folder_paths
from loss_functions import get_combined_loss
from multi_class_unet_model_build import jacard_coef
from image_processing_functions import process_image
from data_augmentation_functions import stack_array
from create_footprint_functions import process_tile, extract_transform_from_directory

# %%
folder_dict = get_folder_paths()
models_dir = Path(folder_dict["models_dir"])
camp_tiles_dir = Path(folder_dict["camp_tiles_dir"])
footprints_dir = Path(folder_dict["footprints_dir"])

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
# ### Move the polygon tiles into a separate folder
#
# > want to experiment with just using polygons not tiles hence keeping these for now

# %%
polygon_directory = area_dir / "polygon_tiles"
polygon_directory.mkdir(exist_ok=True)

for file in area_dir.glob("*.tif"):
    if "polygons" in file.stem:
        destination_path = polygon_directory / file.name
        shutil.move(file, destination_path)
        print(f"Moved {file.name} to {polygon_directory}")

# %% [markdown]
# ### Get image crs

# %%
tiff_files = list(area_dir.glob("*.tif"))
if not tiff_files:
    print("No GeoTIFF files found in the directory.")
    exit()

# Read the first GeoTIFF file
first_tiff_file = tiff_files[0]

# Open the GeoTIFF file
with rio.open(first_tiff_file) as src:
    # Get CRS
    crs = src.crs

print(f"CRS for {area}:", crs)

# %% [markdown]
# ## Process img files
#
# > Put `.geotiff` through same process as training images and output as `.npy`

# %%
# list all .tif files in directoy
img_files = list(area_dir.glob("*.tif"))
img_size = 384

# %%
error_files = []

for img_file in img_files:
    try:
        process_image(img_file, img_size, area_dir)
    except Exception as e:
        print(f"Error processing file {img_file}: {e}")
        error_files.append(img_file)

print("Files with errors:", error_files)


# %% [markdown]
# ### Create stacked arrays and delete `.npy` files in directory

# %%
unseen_images, unseen_filenames = stack_array(area_dir, expanded_outputs=True)
print(unseen_images.shape)
print(unseen_filenames.shape)

# %%
# for file in Path(area_dir).glob("*npy"):
# file.unlink()

# %% [markdown]
# ### Add padding

# %%
padding = 8

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
def is_array_empty(arr):
    return np.all(arr == 0)


empty_indices = [i for i, arr in enumerate(padded_unseen_images) if is_array_empty(arr)]

unseen_filtered = np.delete(padded_unseen_images, empty_indices, axis=0)
filenames_filtered = [
    filename for i, filename in enumerate(unseen_filenames) if i not in empty_indices
]

print("Filtered stacked array shape:", unseen_filtered.shape)
print("Number of filtered filenames:", len(filenames_filtered))


# %% [markdown]
# ### Clear memory

# %%
unseen_images = []
padded_unseen_images = []
unseen_filenames = []

# %% [markdown]
# ## Load model

# %%
# check total loss has loaded successfully
total_loss = get_combined_loss()

# %%
model_path = models_dir / "qa_testing_2024-03-27_0955.hdf5"

model = load_model(
    model_path,
    custom_objects={
        "dice_loss_plus_1focal_loss": total_loss,
        "dice_loss_plus_focal_loss": total_loss,
        "focal_loss": total_loss,
        "dice_loss": total_loss,
        "jacard_coef": jacard_coef,
    },
)

# %% [markdown]
# ## Run model on unseen images (optional)

# %%
predictions = model.predict(unseen_filtered)

# %%
predictions.shape

# %% [markdown]
# ### Visual check

# %%
test_number = 20

# %%
img_size = 384
image_test = np.load(area_dir.joinpath(f"{filenames_filtered[test_number]}.npy"))
image_test.shape

# %%
# BGR to RGB
image_test = image_test[:, :, :3]
image_test = image_test[:, :, ::-1]

# %%
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title(filenames_filtered[test_number])
plt.imshow(image_test)  # [:, :, :3])
plt.subplot(232)
plt.imshow(predictions[test_number])
plt.show()

# %% [markdown]
# ## Convert to polygons

# %%
# get transformation matrix from original .tiff images
transforms = extract_transform_from_directory(area_dir)

# %% jupyter={"outputs_hidden": true}
# create georeferenced footpritns
num_classes = 3
unique_classes = list(range(num_classes))
all_results = []
for idx, (tile, filename) in enumerate(zip(unseen_filtered, filenames_filtered)):
    result_gdf = process_tile(
        model, tile, unique_classes, filename, idx, transforms, crs
    )
    if result_gdf is not None:
        all_results.append(result_gdf)

all_polygons_gdf = pd.concat(all_results, ignore_index=True)


# %% [markdown]
# ### Save polygons for outputting

# %%
output_footprints = footprints_dir / f"{folder_dropdown.value}_footprints.geojson"
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
filtered_gdf = all_polygons_gdf[all_polygons_gdf["index_num"] == 20]

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
