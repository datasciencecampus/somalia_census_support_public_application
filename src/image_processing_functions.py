""" Script of functions related to image preprocessing. """


import warnings

import numpy as np
import rasterio as rio
from sklearn.preprocessing import MinMaxScaler


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

    # Converts banded image into a single column
    # img_arr.shape[0] used to count number of bands
    ascolumns = img_arr.reshape(-1, img_arr.shape[0])

    norm_ascolumns = np.array(
        [
            min_max_scaler.fit_transform(
                clip_to_soft_min_max(ascolumns, clipping_percentile_range)
            )
        ]
    )
    normalised_img = norm_ascolumns.reshape(img_arr.shape)
    return normalised_img


def reorder_array(img_arr, height_index, width_index, bands_index):
    # Re-order the array into height, width, bands order.
    arr = np.transpose(img_arr, axes=[height_index, width_index, bands_index])
    return arr


def check_img_files(img_dir, ref_shape=(384, 384, 4)):
    """
    Check all .npy files in the given directory against a reference shape.

    Args:
    img_dir(str or pathlib.Path): Path to the directory containing the image files.
    ref_shape (tuple of int, optional): The reference shape that each image should have.
        Defaults to (384, 384, 4).

    Raises:
        Warning: If an image file has a different shape than the reference shape.
    """
    img_file_list = img_dir.glob("*npy")
    for file in img_file_list:
        img_array = np.load(file)
        if img_array.shape != ref_shape:
            warnings.warn(f"{file} has a different shape than the reference shape")
