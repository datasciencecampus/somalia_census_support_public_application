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
priority_area_of_interest = "Baidoa"
path_to_imgs = planet_imgs_path.joinpath(priority_area_of_interest)
check_zipped_dirs_and_unzip(path_to_imgs)

observation_path_list = list_directories_at_path(path_to_imgs)

tiff_img_list = get_raster_list_for_given_area(observation_path_list)

observation_dates = [
    extract_dates_from_image_filenames(file_name.stem) for file_name in tiff_img_list
    ]

raster_filepath = tiff_img_list[0]

img_array = return_array_from_tiff(raster_filepath)

img_arr_reordered = change_band_order(img_array)

normalised_img = clip_and_normalize_raster(img_arr_reordered, 99)

# %%
doolow = [4.160722262, 42.0770588]

# %%
m = folium.Map(location=doolow)

# %% [markdown]
# ### Add ESRI satellite imagery as layer

# %%
tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)

# %% [markdown]
# ### Add raster layer

# %%
folium.raster_layers.ImageOverlay(normalised_img.transpose(1, 2, 0),
                                  bounds = get_reprojected_bounds(raster_filepath),
                                  name="Baidoa Planet raster",
                                  interactive=True,
                                 ).add_to(m)


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
# ### Add UNFPA priority extents as layer

# %%
priority_extents = folium.FeatureGroup(name="priority_extents")

shapefiles_dir = data_dir.joinpath(
        "IDP Priority Area Extent Shapefiles",
        "IDP Priority Area Extent Shapefiles",
        "IDP Survey Shapefiles",
)

for area in priority_areas:
    shapefile_full_path = shapefiles_dir.joinpath(f"{area}_Extent.shp")
    check_crs_and_reset(shapefile_full_path)
    convert_shapefile_to_geojson(
        shapefile_full_path,
        priority_area_geojsons_dir,
        )
    extents_path = priority_area_geojsons_dir.joinpath(f"{area}_extent.geojson")
    area_of_interest = json.load(open(extents_path))

    folium.GeoJson(area_of_interest, name=f"{area} UNFPA extent").add_to(priority_extents)

# %%
priority_extents.add_to(m)

# %% [markdown]
# ### Add DSC training data as layer

# %%
training_data = gpd.read_file(data_dir.joinpath("training_data.geojson"))

# %%
folium.GeoJson(training_data, name=f"Doolow labelled training data").add_to(m)

# %% [markdown]
# ## Display map

# %%
folium.LayerControl().add_to(m)

# %%
m


# %% [markdown]
# ## Process training data into raster

# %%
images_dir = data_dir.joinpath("20220830_070622_Dolow_skysatcollect_pansharpened_udm2", "files")
raster_file_path = images_dir.joinpath("20220830_070622_ssc2_u0001_pansharpened_clip.tif")

building_class_list = ["House", "Tent", "Service"]

segmented_training_arr = rasterize_training_data(training_data, raster_file_path, building_class_list, "test3.tif")

# %%
import matplotlib.pyplot as plt
plt.imshow(segmented_training_arr)

# %%
