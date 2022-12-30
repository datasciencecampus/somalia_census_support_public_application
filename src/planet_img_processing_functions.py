from pathlib import Path
from osgeo import gdal
from datetime import datetime
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.plot import show
from sklearn.preprocessing import MinMaxScaler

from functions_library import setup_sub_dir, generate_file_list

def extract_dates_from_image_filenames(file_name):
    """
    Extract date information from Planet imagery file names.

    e.g. file named 'Doolow_W_50MB_20221101' would return
    datetime.datetime(2022, 11, 1, 0, 0)

    Parameters
    ----------
    file_name : str
        The file name for given Planet image.

    Returns
    -------
    datetime.datetime
        Date data type from string.
    """
    numerical_string = file_name.split("_")[-1]
    date = datetime.strptime(numerical_string, "%Y%m%d")
    return date


def unzip_image_files(file_to_unzip):
    # opening the zip file in READ mode
    with ZipFile(file_to_unzip, 'r') as zip:
        # extracting all the files
        zip.extractall(setup_sub_dir(file_to_unzip.parent, file_to_unzip.stem))
        print(f"Files extracted {file_to_unzip.parent.joinpath(file_to_unzip.stem)}")


data_dir = Path.cwd().parent.joinpath("data")

planet_imgs_path = setup_sub_dir(data_dir, "planet_images")

path_to_imgs = planet_imgs_path.joinpath("Doolow")

zipped_observation_list = generate_file_list(
        path_to_imgs, "zip", []
    )

zipped_files = [file_name.stem for file_name in zipped_observation_list]

observation_dates = [
    extract_dates_from_image_filenames(file_name) for file_name in zipped_files
    ]

for observation in zipped_observation_list:
    dir_path = observation.parent.joinpath(observation.stem)

    # if directory does not exist, unzip it first
    if not dir_path.is_dir():
        unzip_image_files(observation)

    tiff_img_list = generate_file_list(
            dir_path.joinpath("files"), "tif", ["pansharpened_clip"]
            )

    planet_img = gdal.Open(str(tiff_img_list[0].resolve()))

# %%

# Refactor into funcs for: (not in order)
# 1) reading planet image
# 2) getting dates from title
# 3) change band order of images
# 4) unzip images
# 5) image standardisation

#%%

def return_array_from_tiff(img_path):
    """Get array from tiff raster.

    Parameters
    ----------
    img_path : Path
        Full path to img file to open.
    """
    with rio.open(img_path) as img:
        img_array = img.read()
    return(img_array)


def change_band_order(
    img_array,
    correct_band_order = [3, 2, 1, 4]
):
    """Changes the order of the raster bands. Default is to assume
    raster order BGR and correct to RGB.

    Parameters
    ----------
    img_array : numpy.ndarray
        The raster in numpy array format.
    correct_band_order : list, optional
        The correct order of bands to get RGB. Planet imagery by default is
        blue, green, red, IR. So we need to parse [3, 2, 1, 4]. This is the
        value.
    """
    img_array = [img_array[band-1] for band in correct_band_order]
    return(np.array(img_array))
# %%
img_array = return_array_from_tiff(tiff_img_list[0])

img_arr_reordered = change_band_order(img_array)

#%%
def return_percentile_range(img_arr, range):
    non_zero_img_arr = img_arr[img_arr>0]
    lower_percentile = np.percentile(non_zero_img_arr, 100-range)
    upper_percentile = np.percentile(non_zero_img_arr, range)
    return(lower_percentile, upper_percentile)

def clip_to_soft_min_max(img_arr, range):
    soft_min, soft_max = return_percentile_range(img_arr, range)
    img_arr_clipped = np.clip(img_arr, soft_min, soft_max)
    return(img_arr_clipped)

#%%

min_max_scaler = MinMaxScaler()

clipping_percentile_range = 99
normalised_img = np.array([
    min_max_scaler.fit_transform(clip_to_soft_min_max(band_array, clipping_percentile_range)) for band_array in img_arr_reordered
    ])
# %%
show(normalised_img)
