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
from pathlib import Path
from pyproj import Transformer 
import rasterio as rio


data_dir = Path.cwd().parent.joinpath("data")

doolow = [4.160722262, 42.0770588]

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

# +
m = folium.Map(location=doolow, zoom_start=13)

# Overlay raster (RGB) called img using add_child() function (opacity and bounding box set)
m.add_child(folium.raster_layers.ImageOverlay(img.transpose(1, 2, 0), 
                                              opacity=0.9,
                                              bounds = bounds_fin
                                             ))

# Display map 
m
