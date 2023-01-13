# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# import standard and third party libraries 
import folium
import json
from pathlib import Path
import geopandas as gpd

# %%
# import custom functions
from functions_library import (
    setup_sub_dir,
    list_directories_at_path
)

from planet_img_processing_functions import (
    check_zipped_dirs_and_unzip,
    extract_dates_from_image_filenames,
    get_raster_list_for_given_area,
    return_array_from_tiff,
    change_band_order,
    clip_and_normalize_raster,
)

from geospatial_util_functions import (
    convert_shapefile_to_geojson,
    get_reprojected_bounds,
    check_crs_and_reset
)

from modelling_preprocessing import rasterize_training_data

# %% [markdown]
# ### Set-up filepaths

# %%
data_dir = Path.cwd().parent.joinpath("data")
planet_imgs_path = setup_sub_dir(data_dir, "planet_images")
priority_area_geojsons_dir = setup_sub_dir(data_dir, "priority_areas_geojson")

# %% [markdown]
# ### Preprocess Planet raster

# %%
priority_areas = ["Doolow",
                  "Mogadishu",
                  "Baidoa",
                  "BeletWeyne",
                  "Bossaso",
                  "Burao",
                  "Dhuusamarreeb",
                  "Gaalkacyo",
                  "Hargeisa",
                  "Kismayo"
                  ]

# %% [markdown]
# ### Load DSC training data 

# %%
training_data = gpd.read_file(data_dir.joinpath("training_data.geojson"))

# %% [markdown]
# ## Process training data into raster

# %%
raster_file_path = data_dir.joinpath("training_tile1.tif")

building_class_list = ["House", "Tent", "Service"]

segmented_training_arr = rasterize_training_data(training_data, raster_file_path, building_class_list, "test3.tif")

# %%
import matplotlib.pyplot as plt
plt.imshow(segmented_training_arr)

# %%
