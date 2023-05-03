""" Script for data augmentation functions """

from pathlib import Path

import numpy as np


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

    # create a horizonatl mirror of each image and stack along the same axis
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
