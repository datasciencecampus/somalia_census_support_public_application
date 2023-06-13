""" Script for data augmentation functions """

from pathlib import Path

import numpy as np
import colorsys


def stack_array(directory, validation_area=None):
    """

    Stack all .npy files in the specified directory (excluding files ending with "background.npy"),
    along with their rotated and mirrored versions, and return the resulting array.

    Args:
        directory (str or Path): The directroy containing the .npy files to stack.
        validation_area (str, optional): The word to exclude from file names. Defaults to None.

    Returns:
        np.ndarray: The stacked array of images/masks.

    """

    # get all .npy files in the directory exlcuding background
    array_files = [
        file
        for file in Path(directory).glob("*npy")
        if not file.name.endswith("background.npy")
        and (validation_area is None or validation_area not in file.name)
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


def stack_array_with_validation(directory, validation_area):
    """
    Stack all .npy files in the specified directory that contain the validation area.

    Args:
        directory (str or Path): The directroy containing the .npy files to stack.
        excluded_word (str, optional): The word to include from file names.

    Returns:
        np.ndarray: The stacked array of images/masks.

    """

    # get all .npy files in the directory exlcuding background
    array_files = [
        file
        for file in Path(directory).glob("*npy")
        if validation_area in file.name and not file.name.endswith("background.npy")
    ]

    # sort the file names alphabetically
    array_files = sorted(array_files)

    # empty list for appending originals
    array_list = []

    # load each .npy and appent to list
    for file in array_files:
        np_array = np.load(file)
        array_list.append(np_array)

    # stack the arrays
    stacked_images = np.stack(array_list, axis=0)

    return stacked_images


def hue_shift(images, shift):
    """
    Apply hue shift to an array of stacked images with RGBN channels

    Args:
        images (np.ndarray): Array of stacked images with shape (n, h, w, 4),
                                     where n is the number of images, h is the height,
                                     w is the weidth, and 4 represents RGBN channels.
        hue_shift (float): The amount of hue shift to apply in the range [0, 1].

    Returns:
    np.adarray: Array of hue_shifted images with the same shape as the input array.
    """
    # perform the hue shift on the stacked_images
    hue_shifted = np.zeros_like(images)

    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            for k in range(images.shape[2]):
                rgbn_pixel = images[i, j, k, :]
                rgb_pixel = rgbn_pixel[:3]
                nir_pixel = rgbn_pixel[3]

                hsv_pixel = colorsys.rgb_to_hsv(
                    rgb_pixel[0], rgb_pixel[1], rgb_pixel[2]
                )
                hsv_pixel = (hsv_pixel[0], (hsv_pixel[1] + shift) % 1, hsv_pixel[2])
                rgb_pixel = colorsys.hsv_to_rgb(
                    hsv_pixel[0], hsv_pixel[1], hsv_pixel[2]
                )

                hue_shifted[i, j, k, :3] = np.array(rgb_pixel)
                hue_shifted[i, j, k, 3] = nir_pixel

    return hue_shifted


def adjust_brightness(images, factor):
    adjusted_images = np.copy(images)
    adjusted_images[..., :3] *= factor

    return adjusted_images


def adjust_contrast(images, factor):
    adjusted_images = np.copy(images)
    # scales the pixel values around a neutral point of 0.5 to effectively preserve the overall brightness while adjusting contrast
    adjusted_images[..., :3] *= (adjusted_images[..., :3] - 0.5) * factor + 0.5

    return adjusted_images


def stack_images(
    stacked_images,
    background_images,
    adjusted_hue,
    adjusted_brightness,
    adjusted_contrast,
    include_hue_adjustment,
    include_backgrounds,
    include_brightness_adjustments,
    include_contrast_adjustments,
):
    """Combine the different augementation stacks based on conditionals"""
    all_stacked_images = stacked_images
    if include_backgrounds:
        all_stacked_images = np.concatenate(
            [all_stacked_images] + [background_images], axis=0
        )
    if include_hue_adjustment:
        all_stacked_images = np.concatenate(
            [all_stacked_images] + [adjusted_hue], axis=0
        )
    if include_brightness_adjustments:
        all_stacked_images = np.concatenate(
            [all_stacked_images] + [adjusted_brightness], axis=0
        )
    if include_contrast_adjustments:
        all_stacked_images = np.concatenate(
            [all_stacked_images] + [adjusted_contrast], axis=0
        )
    return all_stacked_images
