""" Script for data augmentation functions """

from pathlib import Path

import numpy as np
from skimage.color import hsv2rgb, rgb2hsv, rgba2rgb


def stack_array(directory):
    """

    Stack all .npy files in the specified directory (excluding files ending with "background.npy"),
    along with their rotated and mirrored versions, and return the resulting array.

    Args:
        directory (str or Path): The directroy containing the .npy files to stack.

    Returns:
        np.ndarray: The stacked array of images/masks.

    """

    # get all .npy files in the directory exlcuding background
    array_files = [
        file
        for file in Path(directory).glob("*npy")
        if not file.name.endswith("background.npy")
    ]

    # sort the file names alphabetically
    array_files = sorted(array_files)

    # empty list for appending originals
    array_list = []

    # load each .npy and append to list
    for file in array_files:
        np_array = np.load(file)
        array_list.append(np_array)

    # create a rotated version of each array and stack along the same axis
    rotations = []
    for i in range(4):
        rotated = np.rot90(array_list, k=1, axes=(1, 2))
        if i > 0:
            rotated = np.fliplr(rotated)
        rotations.append(rotated)

    # create a horizontal mirror of each image and stack along the same axis
    mirrors = [
        np.fliplr(array_list),
        np.fliplr(rotations[0]),
        np.fliplr(rotations[1]),
        np.fliplr(rotations[2]),
    ]

    # stack the original arrays, rotated versions and mirror versions
    stacked_images = np.concatenate([array_list] + rotations + mirrors, axis=0)

    return stacked_images


def stack_background_arrays(directory):
    """
    Load all .npy files ending with 'background.npy' in the specified directory,
    then sort alphabetically, and return a list of the loaded arrays.

    Args:
        directory (str or Path): The directory containing the .npy files to load.

    Returns:
        List(np.ndarray]: The list of loaded arrays.

    """
    # get all .npy files ending with 'background.npy' in the directory
    background_files = [file for file in Path(directory).glob("*background.npy")]

    # sort the file names alphabetically
    background_files = sorted(background_files)

    # empty list for appending loaded arrays
    background_arrays = []

    # load each .npy and append to list
    for file in background_files:
        np_array = np.load(file)
        background_arrays.append(np_array)
    return background_arrays


def hue_shift_with_alpha(image, shift_value):
    """
    Perform hue shift on an RGBA image while preserving the alpha channel.

    Parameters:
    image (np.ndarray0: Input RGBA image with shape (height, width, 4).
    shift_value (float): Hue shift value to be added to the hue channel (range: [0,1])

    Returns:
    np.adarray: Hue-shifted image with preserved alpha channel, shape (height, width, 4).
    """

    # convert the image from RGBA to RGB color space
    rgb_image = rgba2rgb(image)

    # convert the image from RGB to HSV color space
    hsv_image = rgb2hsv(rgb_image)

    # extract the hue channel from the HSV image
    hue_channel = hsv_image[:, :, 0]

    # perform the hue shift (add a constant value to the hue channel)
    hue_shifted = (hue_channel + shift_value) % 1.0

    # update the hue channel in the HSV image with the shifted values
    hsv_image[:, :, 0] = hue_shifted

    # convert the image back to RGB color space
    rgb_shifted_image = hsv2rgb(hsv_image)

    # create a new array by combining the RGB shifted image and the alpha channel
    shifted_image = np.concatenate(
        (rgb_shifted_image, image[:, :, 3][:, :, np.newaxis]), axis=2
    )

    return shifted_image
