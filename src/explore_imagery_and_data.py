# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# installed
# !pip install rasterio

# %%
#added
import rasterio as rio

# %%
# import standard and third party libraries 
import folium
import json
from pathlib import Path
import geopandas as gpd
import pandas as pd

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
                                 ).add_to(m);


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
priority_extents.add_to(m);

# %% [markdown]
# ### Add DSC training data as layer

# %%
training_data = gpd.read_file(data_dir.joinpath("training_data.geojson"))

# %%
folium.GeoJson(training_data, name=f"Doolow labelled training data").add_to(m);

# %% [markdown]
# ## Add survey IDP location dataframe

# %%
#load in dataframe with locations
survey_idp_locations = pd.read_csv(data_dir.joinpath("Q3_Coordinates_Only_Master_List.csv"))

# %%
#check dataframe has loaded
survey_idp_locations.head()

# %%
#set index
survey_idp_locations.set_index("CCCM IDP Site Code", inplace=True)

# %%
survey_idp_locations.tail()

# %% [markdown]
# ## Check for null values

# %%
survey_idp_locations.isna().sum()

# %%
#drop columns to leave IDP site, long and lat
idp_coordinates=survey_idp_locations.drop(["Region","District", "Neighbourhood", "Neighbourhood Type", "Date IDP site Established", "Source(Q3-2022)", "Comments(Q3-2022)"], axis=1)

# %%
idp_coordinates.head()

# %%
#drop 34 missing values for lat and long
idp_coordinates=idp_coordinates.dropna()

# %%
#check no null values
idp_coordinates.isna().sum()

# %% [markdown]
# ## Separate IDP site, latitude and longitude

# %%
#create nested list to be used in for loop
site_coord_list=idp_coordinates[['IDP Site','Latitude','Longitude']].values.tolist()

# %%
site_coord_list

# %% [markdown]
# ## Add layer using feature group and add child

# %%

feature_group=folium.FeatureGroup(name=("site of interest"))

for i in site_coord_list:
    feature_group.add_child(folium.Marker(location=[i[1],i[2]], popup=i[0], icon=folium.Icon(color="red")))
m.add_child(feature_group)



# %%
folium.LayerControl().add_to(m);

# %%
m
