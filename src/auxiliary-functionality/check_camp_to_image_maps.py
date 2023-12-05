"""Script aimed at comparing the coverage of known camps within existing stores of Planet imagery."""

#%%
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer
import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm


def check_coords_in_raster(
    planet_raster_file_list: list,
    camps_df: pd.DataFrame,
    camps_data_crs: str = "EPSG:4326",
):
    """
    Check whether coordinates in camps_df are within any rasters within planet_raster_file_list.

    For each raster, the pixel value at for each given coordinate is returned. If
    a valid pixel value is found, then the coordinate is said to be within the
    raster and the file path to the raster is appended to the camps_df dataframe.

    Parameters
    ----------
    planet_raster_file_list : list
        List of file paths to the raster files to be checked.
    camps_df : pandas.DataFrame
        Table of known camps, including coordinates and other information.
    camps_data_crs : str, optional
        The CRS of the coordinates within camps_df. By default this is "EPSG:4326"
        i.e. longitude and latitude.

    Returns
    -------
    pandas.DataFrame
        camps_df dataframe with the column "planet_images" updated to include
        the file paths to any rasters containing the given camp on each row
    """
    for raster in tqdm(planet_raster_file_list):
        print(f"Checking camps coords within {raster}")

        with rio.open(raster) as src:
            meta = src.meta
            no_data_value = src.nodata

            for index, row in camps_df.iterrows():
                # My target coordinates
                y_coord = row.Latitude
                x_coord = row.Longitude

                # transform coordinates from long-lat to CRS of raster
                transformer = Transformer.from_crs(
                    camps_data_crs, src.crs, always_xy=True
                )
                xx, yy = transformer.transform(x_coord, y_coord)

                # transform geographic coordinates to raster pixel coordinates
                raster_xy = rio.transform.rowcol(meta["transform"], xs=xx, ys=yy)

                # Load specific pixel only using a window
                window = Window(raster_xy[1], raster_xy[0], 1, 1)
                arr = src.read(window=window)

                # If position not in raster, then arr empty and shape dimension = 0.
                # If position in raster bounding box, but not within image extent,
                # i.e. coords are in the border, then pixel value should be 0 (nodata value).
                if arr.shape[1] > 0 and np.nanmean(arr) > no_data_value:
                    file_name = str(raster).split("planet_images")[1]
                    file_column = camps_df.loc[index, "planet_images"]
                    if file_column == "NONE":
                        camps_df.loc[index, "planet_images"] = file_name
                    else:
                        camps_df.loc[index, "planet_images"] = (
                            file_column + ", " + file_name
                        )
    return camps_df


script_path = Path(__file__).resolve().parent.parent

# == Update here to reflect local file structure (with dirs relative to src) ==#
data_dir = script_path.parent.joinpath("data")
planet_images_file_path = data_dir.joinpath("planet_images")
idp_camps_data_file_path = data_dir.joinpath("somalia_idp_sites_march23.xlsx")
# =====================================================#

planet_raster_file_list = [*planet_images_file_path.rglob("*pansharpened_clip.tif")]

camps_df = pd.read_excel(idp_camps_data_file_path)
camps_df["planet_images"] = "NONE"

# %%
camps_df = check_coords_in_raster(planet_raster_file_list, camps_df)

# %%
# Display summary statistics

## Camp numbers
camps_df.planet_images.value_counts().div(camps_df.planet_images.count()).mul(100)

## Individual camp dweller numbers
camps_df.groupby("planet_images")[" Individual (Q1-2023) "].sum().div(
    camps_df[" Individual (Q1-2023) "].sum()
).mul(100)

# %%
# Save output dataframe
camps_df.to_csv(
    data_dir.joinpath("somalia_idp_sites_planet_imagery_coverage_march23.csv")
)
