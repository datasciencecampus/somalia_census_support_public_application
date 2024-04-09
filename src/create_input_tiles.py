#!/usr/bin/env python
# coding: utf-8
# %%
from rasterio.mask import mask
from pathlib import Path
import geopandas as gpd
import rasterio as rio
from functions_library import generate_tiles

# %% [markdown]
# #### Set directories

# %%
# set directories - note this is local so yours might look different!
data_dir = Path.cwd().parent.joinpath("data")
planet_images_dir = data_dir.joinpath("planet_images")  # might not be necessary

baidoa_dir = planet_images_dir.joinpath("Baidoa")
tile_dir = baidoa_dir.joinpath("tiles")


# %% [markdown]
# #### Import raster and geojson

# %%
baidoa_holwadag_img = baidoa_dir.joinpath(
    "20230302_064032_ssc2_u0001_pansharpened_clip.tif"
)
baidoa_holwaday_polygons = baidoa_dir.joinpath("baidoa_holwadag_camp_extents.geojson")

# %% [markdown]
# ## Creating polygons & tiles for input

# %% [markdown]
# ### Create clipped polygons

# %%
polygons_gdf = gpd.read_file(baidoa_holwaday_polygons)

# this creates clipped rasters of the larger polygons - import these into GCP as well as I want to see if we can just run these
with rio.open(baidoa_holwadag_img) as src:
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

            output_file = tile_dir / f"baidoa_holwadag_polygons_{idx}.tif"

            with rio.open(output_file, "w", **out_meta) as dst:
                dst.write(out_image)

        else:
            print(f"Skipping empty geometry for index {idx}")


# %% [markdown]
# ### Create tiles

# %%
for tile_path in tile_dir.glob("*.tif"):
    generate_tiles(tile_path, output_dir=tile_dir)
