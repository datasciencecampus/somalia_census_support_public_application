#!/usr/bin/env python
# coding: utf-8
# %%
from rasterio.mask import mask
from pathlib import Path
import geopandas as gpd
import rasterio as rio
import ipywidgets as widgets
from IPython.display import display
from functions_library import generate_tiles

# %% [markdown]
# #### Set directories

# %%
# set directories - note this is local so yours might look different!
data_dir = Path.cwd().parent.joinpath("data")

# change variable for dir name as required
region_dir = data_dir.joinpath("Baidoa")
tile_dir = region_dir.joinpath("tiles")


# %% [markdown]
# #### Import raster and geojson

# %%
pattern = "*_camp_extents.geojson"
matching_files = list(region_dir.glob(pattern))

# %%
folder_dropdown = widgets.Dropdown(options=matching_files, description="select folder:")
display(folder_dropdown)

# %%
if folder_dropdown.value.stem == "baidoa_central_camp_extents":
    planet_file = "20230320_104520_ssc11_u0001_pansharpened_clip.tif"

elif folder_dropdown.value.stem == "baidoa_east_camp_extents":
    planet_file = "20230320_064000_ssc2_u0001_pansharpened_clip.tif"

elif folder_dropdown.value.stem == "baidoa_west_camp_extents":
    planet_file = "20230113_104649_ssc6_u0001_pansharpened_clip.tif"

elif folder_dropdown.value.stem == "baidoa_holwadag_camp_extents":
    planet_file = "20230302_064032_ssc2_u0001_pansharpened_clip.tif"

# baidoa_ishaholwada
else:
    planet_file = "20230302_064032_ssc2_u0001_pansharpened_clip_cloudy.tif"

# %%
# tif file
img_file = region_dir.joinpath(planet_file)

# geojson file
polygon_file = region_dir.joinpath(f"{folder_dropdown.value.stem}.geojson")

# %% [markdown]
# ## Creating polygons & tiles for input

# %%
polygons_gdf = gpd.read_file(polygon_file)

# this creates clipped rasters of the larger polygons - import these into GCP as well as I want to see if we can just run these
with rio.open(img_file) as src:
    for idx, polygon in polygons_gdf.iterrows():
        if not polygon.geometry.is_empty:
            out_image, out_transform = mask(src, [polygon.geometry], crop=True)

            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

            output_file = tile_dir / f"{folder_dropdown.value.stem}_polygons_{idx}.tif"

            with rio.open(output_file, "w", **out_meta) as dst:
                dst.write(out_image)

        else:
            print(f"Skipping empty geometry for index {idx}")

# %% [markdown]
# ### Create clipped polygons

# %% [markdown]
# ### Create tiles

# %%
for tile_path in tile_dir.glob("*_polygons_*.tif"):
    generate_tiles(tile_path, output_dir=tile_dir)
