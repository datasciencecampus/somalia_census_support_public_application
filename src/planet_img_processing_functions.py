#%%

from datetime import datetime
from zipfile import ZipFile

import numpy as np
import rasterio as rio
from sklearn.preprocessing import MinMaxScaler

from functions_library import generate_file_list, setup_sub_dir


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
    numerical_string = file_name.split("_")[0]
    date = datetime.strptime(numerical_string, "%Y%m%d")
    return date


def unzip_image_files(file_to_unzip):
    """Extract content from zipped folder into dir of same name at same location.

    Parameters
    ----------
    file_to_unzip : path
        Path to zipped folder to extract.
    """
    with ZipFile(file_to_unzip, "r") as zip:
        # extracting all the files
        zip.extractall(setup_sub_dir(file_to_unzip.parent, file_to_unzip.stem))
        print(f"Files extracted {file_to_unzip.parent.joinpath(file_to_unzip.stem)}")


def check_zipped_dirs_and_unzip(path_to_imgs):
    """Check image directory and if only zipped files present, extract them.

    Parameters
    ----------
    path_to_imgs : path
        Path to subdirectory in data where satellite rasters stored for given location.
    """
    zipped_observation_list = generate_file_list(path_to_imgs, "zip", [])

    for observation in zipped_observation_list:
        dir_path = observation.parent.joinpath(observation.stem)

        # if directory does not exist, unzip it first
        if not dir_path.is_dir():
            print("Unzipping zipped raster folder.\n")
            unzip_image_files(observation)


def get_raster_list_for_given_area(observation_path_list):
    tiff_img_list = []
    for directory in observation_path_list:
        tiff_img = generate_file_list(
            directory.joinpath("files"), "tif", ["pansharpened_clip"]
        )
        tiff_img_list.append(tiff_img[0])
    return tiff_img_list


def return_array_from_tiff(img_path):
    """Get array from tiff raster.

    Parameters
    ----------
    img_path : Path
        Full path to img file to open.
    """
    with rio.open(img_path) as img:
        img_array = img.read()
    return img_array


def change_band_order(img_array, correct_band_order=[3, 2, 1, 4]):
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
    img_array = [img_array[band - 1] for band in correct_band_order]
    return np.array(img_array)


def return_percentile_range(img_arr, range):
    """Select pixels with value above zero and return upper and lower percentiles
    for given range. E.g. range = 98 returns the 2% and 98% percentiles.

    Parameters
    ----------
    img_arr : numpy.ndarray
        The array representation of the satellite raster.
    range : float or int
        The range at which to return upper and lower percentiles. A value of 90
        would return the 10th and 90th percentile values.
    """
    non_zero_img_arr = img_arr[img_arr > 0]
    lower_percentile = np.percentile(non_zero_img_arr, 100 - range)
    upper_percentile = np.percentile(non_zero_img_arr, range)
    return (lower_percentile, upper_percentile)


def clip_to_soft_min_max(img_arr, range):
    """Calculate percentile values for given range and clip all values above and below.

    Parameters
    ----------
    img_arr : numpy.ndarray
        The array representation of the satellite raster.
    range : float or int
        The range at which to return upper and lower percentiles. A value of 90
        would return the 10th and 90th percentile values.
    """
    soft_min, soft_max = return_percentile_range(img_arr, range)
    img_arr_clipped = np.clip(img_arr, soft_min, soft_max)
    return img_arr_clipped


def clip_and_normalize_raster(img_arr, clipping_percentile_range):
    """Clip raster by percentile range and then normalise to [0,1] range.

    Parameters
    ----------
    img_arr : numpy.ndarray
        The array representation of the satellite raster.
    clipping_percentile_range : _type_
        The range at which to return upper and lower percentiles. A value of 90
        would return the 10th and 90th percentile values.
    """
    min_max_scaler = MinMaxScaler()
    normalised_img = np.array(
        [
            min_max_scaler.fit_transform(
                clip_to_soft_min_max(band_array, clipping_percentile_range)
            )
            for band_array in img_arr
        ]
    )
    return normalised_img
