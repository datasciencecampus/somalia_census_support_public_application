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
    """
    Adjust the brightness of input images by multiplying RGB channels with a scalar factor.

    Parameters:
        images (numpy.ndarray): Array of input images with shape (batch_size, height, width, channels).
        factor (float): Brightness adjustment factor.

    Returns:
        numpy.ndarray: Array of adjusted images with shape (batch_size, height, width, channels).
    """
    adjusted_images = np.copy(images)

    # Multiply RGB channels with the adjustment factor
    adjusted_images[..., :3] *= factor

    return adjusted_images


def adjust_contrast(images, clip_limit=2.0, tile_grid_size=(8, 8)):

    """
    Adjust the contrast of input images using Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters:
        images (numpy.ndarray): Array of input images with shape (batch_size, height, width, channels).
        clip_limit (float): Threshold for contrast limiting (default is 2.0).
        tile_grid_size (tuple): Size of grid for histogram equalization (default is (8, 8)).

    Returns:
        numpy.ndarray: Array of adjusted images with shape (batch_size, height, width, channels).
    """

    # Create copy array
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
    image_mask = np.copy(image_mask)
    kernel = np.ones((3, 3), np.uint8)

    # Create borders for Buildings
    eroded_building = cv2.erode(
        (image_mask == 1).astype(np.uint8), kernel, iterations=1
    )
    border_building = (image_mask == 1).astype(np.uint8) - eroded_building
    image_mask[(border_building > 0) & (image_mask == 1)] = 1

    # Create borders for Tents
    eroded_tent = cv2.erode((image_mask == 2).astype(np.uint8), kernel, iterations=1)
    border_tent = (image_mask == 2).astype(np.uint8) - eroded_tent
    image_mask[(border_tent > 0) & (image_mask == 2)] = 2

    # Set interior of the polygons to 0 (background)
    image_mask[(image_mask == 1) & (border_building == 0)] = 0
    image_mask[(image_mask == 2) & (border_tent == 0)] = 0

    return image_mask


def create_class_borders_batch(image_masks):
    """
    Create border masks for each mask in the batch.

    Parameters:
        image_masks (numpy.ndarray): Array of binary masks with shape (batch_size, height, width).

    Returns:
        numpy.ndarray: Array of border masks with shape (batch_size, height, width).
    """

    # Set parameters
    batch_size, height, width = image_masks.shape
    # Empty mask array
    result_masks = np.zeros_like(image_masks)

    # Iterate over each mask in the batch and create border masks
    for i in range(batch_size):
        image_mask = image_masks[i]
        result_mask = create_class_borders_array(image_mask)
        result_masks[i] = result_mask

    return result_masks


def split_arrays(images, masks, edges, filenames, overlap_pixels=20):
    """
    Split input images, masks, and edges into smaller tiles.

    Parameters:
        images (numpy.ndarray): Array of input images with shape (batch_size, height, width, channels).
        masks (numpy.ndarray): Array of masks with shape (batch_size, height, width).
        edges (numpy.ndarray): Array of edge masks with shape (batch_size, height, width).
        filenames (list): List of filenames corresponding to the images.
        overlap_pixels (int): Number of overlapping pixels between tiles (default is 20).

    Returns:
        tuple: A tuple containing four numpy arrays representing the split images, masks, edges, and the corresponding filenames.
            - new_images: Array of split images with shape (batch_size * 4, tile_height, tile_width, channels).
            - new_masks: Array of split masks with shape (batch_size * 4, tile_height, tile_width).
            - new_edges: Array of split edge masks with shape (batch_size * 4, tile_height, tile_width).
            - new_filenames: List of filenames corresponding to the split images.
    """

    # Set parameters
    batch_size, height, width, channels = images.shape

    # Reshape filenames to match the new batch size
    new_filenames = []
    for filename in filenames:
        new_filenames.extend(
            [filename + "_a", filename + "_b", filename + "_c", filename + "_d"]
        )

    new_filenames = np.array(new_filenames)

    # Calculate the new tile size
    tile_height = height // 2
    tile_width = width // 2

    # Initialize arrays to store split images, masks, and edges
    new_images = np.empty((batch_size * 4, tile_height, tile_width, channels))
    new_masks = np.empty((batch_size * 4, tile_height, tile_width))
    new_edges = np.empty((batch_size * 4, tile_height, tile_width))

    # Split each image, mask, and edge into four tiles
    for i in range(batch_size):
        for j, suffix in enumerate(["_a", "_b", "_c", "_d"]):
            h_start = max(0, (j // 2) * (tile_height - overlap_pixels))
            h_end = min(height, h_start + tile_height)
            w_start = max(0, (j % 2) * (tile_width - overlap_pixels))
            w_end = min(width, w_start + tile_width)

            new_images[i * 4 + j] = images[i, h_start:h_end, w_start:w_end, :]
            new_masks[i * 4 + j] = masks[i, h_start:h_end, w_start:w_end]
            new_edges[i * 4 + j] = edges[i, h_start:h_end, w_start:w_end]

    return new_images, new_masks, new_edges, new_filenames
