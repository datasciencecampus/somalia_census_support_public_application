""" Script of functions related to image preprocessing. """


import warnings

import numpy as np
import rasterio as rio
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


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
        ],
        dtype="float32",
    )
    normalised_img = norm_ascolumns.reshape(img_arr.shape)
    return normalised_img


def reorder_array(img_arr, height_index, width_index, bands_index):
    # Re-order the array into height, width, bands order.
    arr = np.transpose(img_arr, axes=[height_index, width_index, bands_index])
    return arr


def process_image(img_file, img_size, img_dir):
    """
    Process a geotiff image for inputting into unet model

    Parameters:
    img_file (Path): The input image file path.
    img_size (int): Target size for image processing
    img_dir (Path): Directory to save processed npy file

    """
    # reading in file with rasterio
    img_array = return_array_from_tiff(img_file)

    # clip to percentile
    arr_normalised = clip_and_normalize_raster(img_array, 99)

    # reorder into height, width, band order
    arr_normalised = reorder_array(arr_normalised, 1, 2, 0)

    # re-sizing to img_size (defined above as 384)
    arr_normalised = arr_normalised[0:img_size, 0:img_size, :]

    # create a new filename without bgr
    img_filename = Path(img_file).stem.replace("_bgr", "").replace("_rgb", "")

    # save the NumPy array
    np.save(img_dir.joinpath(f"{img_filename}.npy"), arr_normalised)


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


def remove_bgr_from_filename(img_dir, img_files):
    """
    Rename all .tiff files in a img_dir by removing '_bgr' from the file name

    Args:
    img_dir (str): Path to directory containing the img_files to rename

    """
    # removing _bgr from file name
    for img_file in img_files:
        # file name without extension
        file_name = img_file.stem

        # check if file name contains '_bgr'
        if "_bgr" in file_name:
            # replace '_bgr' with empty string to remove
            new_name = file_name.replace("_bgr", "")

            # create the new file name with .tiff extension
            new_name += ".tif"

            # rename the file with the new name
            img_file.rename(img_dir / new_name)
