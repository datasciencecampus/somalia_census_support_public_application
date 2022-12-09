# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import folium
import geopandas
import json
from pathlib import Path
from pyproj import Transformer 
import rasterio as rio


# import custom functions
from functions_library import setup_sub_dir, create_geojsons_of_extents

data_dir = Path.cwd().parent.joinpath("data")
priority_area_geojsons_dir = setup_sub_dir(data_dir, "priority_areas_geojson")

doolow = [4.160722262, 42.0770588]

m = folium.Map(location=doolow)

tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)

# +
# code largely coutesy of https://gis.stackexchange.com/questions/393938/plotting-landsat-image-on-folium-maps

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

# Finding the centre latitude & longitude    
centre_lon = bounds_fin[0][1] + (bounds_fin[1][1] - bounds_fin[0][1])/2
centre_lat = bounds_fin[0][0] + (bounds_fin[1][0] - bounds_fin[0][0])/2
# -

# Overlay raster (RGB) called img using add_child() function (opacity and bounding box set)
folium.raster_layers.ImageOverlay(img.transpose(1, 2, 0), 
                                  bounds = bounds_fin,
                                  name="Doolow Planet raster"
                                 ).add_to(m)

priority_areas = ["Doolow",
                  "Mogadishu",
                  "Baidoa",
                  "BeletWeyne",
                  "Bossaso",
                  "Burao",
                  "Dhuusamarreeb",
                  "Gaalkacyo", 
                  "Hargeisa",
                  "Kismayo"]

for area in priority_areas:
    create_geojsons_of_extents(area, data_dir)
    extents_path = priority_area_geojsons_dir.joinpath(f"{area}_extent.geojson")
    area_of_interest = json.load(open(extents_path))

    folium.GeoJson(area_of_interest, name=f"{area} UNFPA extent").add_to(m)

# +
folium.LayerControl().add_to(m)

# Display map 
m
# -


