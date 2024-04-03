#!/usr/bin/env python
# coding: utf-8
# %%
import rasterio as rio
from rasterio.mask import mask
from rasterio.windows import Window
import numpy as np
from pathlib import Path
import geopandas as gpd

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


# %% [markdown]
# ### Create tiles

# %%


def generate_tiles(image_path, tile_size=384, output_dir="tiles"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rio.open(image_path) as src:
        rows = src.height // tile_size
        cols = src.width // tile_size

        for row in range(rows):
            for col in range(cols):
                window = Window(col * tile_size, row * tile_size, tile_size, tile_size)
                # Read the tile
                tile = src.read(window=window)

                if np.sum(tile) > 0:
                    tile_meta = src.meta.copy()
                    tile_meta.update(
                        {
                            "width": tile_size,
                            "height": tile_size,
                            "transform": rio.windows.transform(window, src.transform),
                        }
                    )

                    tile_filename = (
                        output_dir
                        / f"{image_path.stem.replace('polygon', '')}_tile_{row}_{col}.tif"
                    )
                    with rio.open(tile_filename, "w", **tile_meta) as dst:
                        dst.write(tile)


# %%
for tile_path in tile_dir.glob("*.tif"):
    generate_tiles(tile_path, output_dir=tile_dir)
