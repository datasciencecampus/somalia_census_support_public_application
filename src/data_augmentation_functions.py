""" Script for data augmentation functions """

from pathlib import Path

import numpy as np
import colorsys
import cv2


def stack_array(directory, expanded_outputs=False):
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
    ]

    # sort the file names alphabetically
    array_files = sorted(array_files)

    # empty list for appending originals
    array_list = []
    # File names for arrays being stacked
    filenames = []

    # load each .npy and append to list
    for file in array_files:
        np_array = np.load(file)
        array_list.append(np_array)
        filenames.append(file.stem)

    # stack the original arrays, rotated versions and mirror versions
    stacked_images = np.concatenate([array_list], axis=0)
    # stacked_images = stacked_images.astype(np.float32)  # Convert to float32
    stacked_filenames = np.concatenate([filenames], axis=0)

    if expanded_outputs:
        return stacked_images, stacked_filenames
    else:
        return stacked_images


def stack_rotate(array_list, filenames, expanded_outputs=False):
    """

    Stack all .npy files in the specified directory (excluding files ending with "background.npy"),
    along with their rotated and mirrored versions, and return the resulting array.

    Args:
        directory (str or Path): The directroy containing the .npy files to stack.
        validation_area (str, optional): The word to exclude from file names. Defaults to None.

    Returns:
        np.ndarray: The stacked array of images/masks.

    """

    # create a rotated version of each array and stack along the same axis
    rotations = []
    for i in range(1, 4):  # Create 3 rotated versions (90, 180, 270 degrees)
        rotated = np.rot90(array_list, k=i, axes=(1, 2))
        rotations.append(rotated)

    # create a horizontal mirror of each image and stack along the same axis
    mirrors = [
        np.fliplr(array_list),
        np.fliplr(rotations[0]),
        np.fliplr(rotations[1]),
        np.fliplr(rotations[2]),
    ]

    # stack the original arrays, rotated versions and mirror versions
    stacked_images = np.concatenate(rotations + mirrors, axis=0)
    # stacked_images = stacked_images.astype(np.float32)  # Convert to float32
    stacked_filenames = np.tile(filenames, 7)

    if expanded_outputs:
        return stacked_images, stacked_filenames
    else:
        return stacked_images


def stack_background_arrays(directory, expanded_outputs=False):
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

    # empty list or appending file names
    background_filenames = []

    # load each .npy and append to list
    for file in background_files:
        np_array = np.load(file)
        background_arrays.append(np_array)
        background_filenames.append(file.stem)

    background_arrays = np.concatenate([background_arrays], axis=0)
    # stacked_images = stacked_images.astype(np.float32)  # Convert to float32
    background_filenames = np.concatenate([background_filenames], axis=0)

    if expanded_outputs:
        return background_arrays, background_filenames
    else:
        return background_arrays


def stack_array_with_validation(directory, validation_area, expanded_outputs=False):
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

    # empty list or appending file names
    val_filenames = []

    # load each .npy and appent to list
    for file in array_files:
        np_array = np.load(file)
        array_list.append(np_array)
        val_filenames.append(file.stem)

    # stack the arrays
    stacked_images = np.stack(array_list, axis=0)

    if expanded_outputs:
        return stacked_images, val_filenames
    else:
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


def adjust_contrast(images, clip_limit=2.0, tile_grid_size=(8, 8)):
    adjusted_images = np.copy(images)

    # Assuming images are in the range [0, 1], convert to uint8 for cv2
    adjusted_images_uint8 = (adjusted_images * 255).astype(np.uint8)

    # Iterate over the images and apply CLAHE to RGB channels
    for i in range(len(adjusted_images_uint8)):
        rgb_image = adjusted_images_uint8[i][:, :, :3]  # Extract RGB channels
        nir_channel = adjusted_images_uint8[i][:, :, 3]  # Extract NIR channel

        # Convert RGB image to LAB color space
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

        # Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel_clahe = clahe.apply(l_channel)

        # Merge the processed L channel with the original A and B channels
        lab_image_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])

        # Convert LAB image back to RGB
        rgb_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2RGB)

        # Combine the adjusted RGB image with the original NIR channel
        adjusted_images_uint8[i] = np.concatenate(
            [rgb_image_clahe, np.expand_dims(nir_channel, axis=-1)], axis=-1
        )

    # Convert back to float in the range [0, 1]
    adjusted_images = adjusted_images_uint8 / 255.0

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


def create_border(image_mask):
    kernel = np.ones((3, 3), np.uint8)

    eroded = cv2.erode((image_mask == 1).astype(np.uint8), kernel, iterations=1)
    border = (image_mask == 1).astype(np.uint8) - eroded
    image_mask[border > 0] = 3

    return image_mask


def create_class_borders(image_mask):
    kernel = np.ones((3, 3), np.uint8)

    # Create borders for Buildings
    eroded = cv2.erode((image_mask == 1).astype(np.uint8), kernel, iterations=1)
    border = (image_mask == 1).astype(np.uint8) - eroded
    image_mask[border > 0] = 3

    # Create borders for Tents
    eroded = cv2.erode((image_mask == 2).astype(np.uint8), kernel, iterations=1)
    border = (image_mask == 2).astype(np.uint8) - eroded
    image_mask[border > 0] = 4

    return image_mask


def process_mask(mask, binary_borders):
    mask_to_update = np.copy(mask)
    test_mask = np.copy(mask_to_update)
    test_mask[test_mask == 2] = 1

    if binary_borders:
        mask_to_update[mask_to_update == 2] = 1
        processed_image_mask = create_border(np.copy(mask_to_update))
        mask_to_update[processed_image_mask == 3] = 3
    else:
        processed_image_mask = create_class_borders(np.copy(mask_to_update))
        mask_to_update[processed_image_mask == 3] = 3
        mask_to_update[processed_image_mask == 4] = 4

    return mask_to_update, test_mask


def create_class_borders_array(image_mask):
    kernel = np.ones((3, 3), np.uint8)
    classes = np.unique(image_mask)

    reduced_classes = image_mask.copy()
    borders = np.zeros_like(image_mask)

    for class_value in classes:
        if class_value == 0:  # skip background
            continue

        eroded = cv2.erode(
            (image_mask == class_value).astype(np.uint8), kernel, iterations=1
        )
        border = (image_mask == class_value).astype(np.uint8) - eroded

        borders[border > 0] = class_value
        reduced_classes[border > 0] = class_value - 2  # reduce the original class by 2

    return reduced_classes, borders
