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
import folium
import json
from pathlib import Path
from pyproj import Transformer
import rasterio as rio
from pathlib import Path
from rasterio.plot import show

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
    create_geojsons_of_extents,
    return_bounds_from_tiff,
    get_reprojected_bounds,
)

# %%
data_dir = Path.cwd().parent.joinpath("data")
planet_imgs_path = setup_sub_dir(data_dir, "planet_images")
priority_area_geojsons_dir = setup_sub_dir(data_dir, "priority_areas_geojson")

# %%
priority_area_of_interest = "Baidoa"
path_to_imgs = planet_imgs_path.joinpath(priority_area_of_interest)
check_zipped_dirs_and_unzip(path_to_imgs)

observation_path_list = list_directories_at_path(path_to_imgs)

tiff_img_list = get_raster_list_for_given_area(observation_path_list)

observation_dates = [
    extract_dates_from_image_filenames(file_name.stem) for file_name in tiff_img_list
    ]

raster = tiff_img_list[0]

img_array = return_array_from_tiff(raster)

img_arr_reordered = change_band_order(img_array)

normalised_img = clip_and_normalize_raster(img_arr_reordered, 99)

#show(normalised_img)

# %%
doolow = [4.160722262, 42.0770588]

# %%
m = folium.Map(location=doolow)

# %%
# Add ESRI satellite tile as a layer into map
tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)

# %%
# code largely courtesy of https://gis.stackexchange.com/questions/393938/plotting-landsat-image-on-folium-maps

# load Planet imagery data for Doolow
# TODO: Generalise for other data files when present.
images_dir = data_dir.joinpath("20220830_070622_Dolow_skysatcollect_pansharpened_udm2", "files")
image_path = images_dir.joinpath("20220830_070622_ssc2_u0001_pansharpened_clip.tif")

dst_crs = "EPSG:4326" # Global projection: WGS84

with rio.open(image_path) as src:
    img = src.read()
    img = img
    src_crs = src.crs["init"].upper()
    min_lon, min_lat, max_lon, max_lat = src.bounds

## Conversion from UTM to WGS84 CRS
bounds_orig = [[min_lat, min_lon], [max_lat, max_lon]]

bounds_fin = []

for item in bounds_orig:
    #converting to lat/lon
    lat = item[0]
    lon = item[1]

    proj = Transformer.from_crs(int(src_crs.split(":")[1]), int(dst_crs.split(":")[1]), always_xy=True)

    lon_n, lat_n = proj.transform(lon, lat)

    bounds_fin.append([lat_n, lon_n])

# %%
# Overlay raster (RGB) called img using add_child() function (opacity and bounding box set)
folium.raster_layers.ImageOverlay(img.transpose(1, 2, 0),
                                  bounds = bounds_fin,
                                  name="Doolow Planet raster"
                                 ).add_to(m)

# %%
folium.raster_layers.ImageOverlay(normalised_img.transpose(1, 2, 0),
                                  bounds = get_reprojected_bounds(raster),
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

# %%
priority_extents = folium.FeatureGroup(name="priority_extents")


for area in priority_areas:
    create_geojsons_of_extents(area, data_dir, priority_area_geojsons_dir)
    extents_path = priority_area_geojsons_dir.joinpath(f"{area}_extent.geojson")
    area_of_interest = json.load(open(extents_path))

    folium.GeoJson(area_of_interest, name=f"{area} UNFPA extent").add_to(priority_extents)

# %%
training_data = json.load(open(data_dir.joinpath("training_data.geojson")))
folium.GeoJson(training_data, name=f"Doolow labelled training data").add_to(m)

# %%
priority_extents.add_to(m)

# %%
folium.LayerControl().add_to(m)

# %% [markdown]
# ## Display map

# %%
m


# %%
