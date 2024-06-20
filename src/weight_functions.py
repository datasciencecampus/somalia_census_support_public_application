""" Script for weights functions """

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from scipy.ndimage import distance_transform_edt
from sklearn.utils.class_weight import compute_class_weight


def calculate_distance_weights(stacked_masks, sigma=3, c=200):
    """
    Calculates average class weights based on an alternative weighting scheme (from google open data).

    Args:
        stacked_masks (np.ndarray): Stacked masks with shape (N, H, W, C)
                                    where N is the number of masks, H is
                                    the height, W is the width, and C
                                    is the number of classes.
        sigma (float): Length scale of the Gaussian kernel. Default is 3.
        c (float): Scaling constant for the weights. Default is 200.

    Returns:
        np.ndarray: Computed average pixel weights with shape (C,).
    """

    num_masks = stacked_masks.shape[0]  # number of masks
    num_classes = stacked_masks.shape[-1]  # number of classes
    weights_sum = np.zeros(num_classes)  # accumalte weights per class

    for i in range(num_masks):
        mask = stacked_masks[i]  # get current mask

        for class_index in range(num_classes):  # iterate over classes
            class_mask = mask[..., class_index]  # get class masks
            boundaries = np.unique(class_mask)  # find unique values in class mask

            for boundary in boundaries:  # iterate over boundaries of class
                boundary_mask = np.where(
                    class_mask == boundary, 1, 0
                )  # create boundary mask
                boundary_edges = np.gradient(
                    boundary_mask.astype(np.float32)
                )  # compute gradient of boundary mask
                weight = np.sum(
                    np.abs(boundary_edges[0]) + np.abs(boundary_edges[1])
                )  # calculate weight of boundary
                weights_sum[class_index] += weight  # accumate weight per class

    avg_weights = (
        gaussian_filter(weights_sum, sigma) * c / num_masks
    )  # apply Gaussian filter and scale by constant
    class_weights = avg_weights / np.sum(avg_weights)  # normlise weights to sum to 1

    return class_weights


def calculate_size_weights(stacked_masks, alpha=1.0):
    """
    Calculates the class weights based on the pixel sizes of the objects.

    Args:
        stacked_masks (np.ndarray): Stacked masks with shape (N, H, W, C)
                                    where N is the number of masks, H is
                                    the height, W is the width, and C
                                    is the number of classes.
        alpha (float): Weighting factor to control the strength of the weight assignment. Default is 1.0.

    Returns:
        np.ndarray: Computed average pixel weights with shape (C,).
    """
    num_masks, height, width, num_classes = stacked_masks.shape
    class_weights = np.zeros(num_classes)

    for i in range(num_masks):
        mask = stacked_masks[i]

        # compute the object sizes for each class
        for j in range(num_classes):
            class_mask = mask[:, :, j]
            labeled_mask, num_objects = label(class_mask)

            # check if any objects are present in the class mask
            if num_objects > 0:
                object_sizes = np.bincount(labeled_mask.flat)[
                    1:
                ]  # count the number of pixels for each polygon

                # calculate the weight based on the pixel size of the object
                weight = np.mean(object_sizes)
                class_weights[j] += weight

    class_weights = alpha / (class_weights + 1e-6)
    class_weights /= np.sum(class_weights)  # normalize the weights

    return class_weights


def calculate_building_distance_weights(stacked_masks, sigma=3, c=200, alpha=1.0):
    """
    Calculate custom class weights that prioritises pixels close to other buildings.

    Args:
        stacked_masks (np.ndarray): Stacked polyon masks with shape (N, H, W, C),
                                    where N is the number of masks, H is the height,
                                    W is the width, and C is the number of classes.
        sigma (float): Length scale of the Gaussian kernel for distance weighting. Default is 3.
        c (float): Scaling constant for the weights. Default is 200.
        alpha (float): Weighting factor to control the strength of the distance weighting. Default is 1.0.

    Returns:
        np.ndarray: Computed class weights with shape (C,).
    """

    num_masks = stacked_masks.shape[0]  # numer of masks
    weights_sum = np.zeros(stacked_masks.shape[-1])  # accumate weights per class

    for i in range(num_masks):
        mask = stacked_masks[i, :, :, :]  # get current mask

        for class_index in range(mask.shape[-1]):  # iterate over classes
            class_mask = mask[:, :, class_index]  # get class mask
            distances = distance_transform_edt(
                class_mask
            )  # calculate Euclidean distance transform
            weights = np.exp(
                -alpha * (distances**2) / (2 * sigma**2)
            )  # calculate distance weights
            weight_sum = np.sum(weights)  # calculate weight sum
            weights_sum[class_index] += weight_sum  # accumulate weight per class

        avg_weights = (
            gaussian_filter(weights_sum, sigma) * c / num_masks
        )  # apply Gaussian filter and scale by constant
        class_weights = avg_weights / np.sum(avg_weights)  # normalise weights to 1

        return class_weights


def calculate_weight_stats(class_weights):
    """
    Calculates the relative differences and ratios of the class weights.

    Args:
        class_weights(np.ndarray): Array of class weights.

    Returns:
        dict: A dictionary containing the relative differences and ratios of the weights.

    """

    weight_stats = {}

    max_weight = np.max(class_weights)
    min_weight = np.min(class_weights)
    weight_range = max_weight - min_weight

    # calculate relative diff
    relative_diffs = (class_weights - min_weight) / weight_range

    # calculate weight ratios
    weight_ratios = class_weights / np.mean(class_weights)

    weight_stats["relative_diffs"] = relative_diffs
    weight_stats["ratios"] = weight_ratios

    return weight_stats


def get_weights(
    folder_dropdown, stacked_masks, stacked_masks_cat, alpha=1.0, sigma=3, c=200
):
    """
    Calculate weights based on different criteria.

    Parameters:
    - folder_dropdown (str): Value selected from dropdown menu, determines the weighting method.
    - stacked_masks (numpy.ndarray): Stacked masks for frequency or building-based weighting.
    - stacked_masks_cat (numpy.ndarray, optional): Stacked categorical masks for Google or size-based weighting.
    - alpha (float, optional): Alpha parameter for size-based weighting.
    - sigma (float, optional): Sigma parameter for building-based weighting.
    - c (float, optional): C parameter for building-based weighting.

    Returns:
    - weights (numpy.ndarray): Computed weights based on the selected method.
    """
    if folder_dropdown == "frequency":
        weights = compute_class_weight(
            "balanced",
            classes=np.unique(stacked_masks),
            y=np.ravel(stacked_masks, order="C"),
        )
    elif folder_dropdown == "google":
        weights = calculate_distance_weights(stacked_masks_cat)
    elif folder_dropdown == "size":
        weights = calculate_size_weights(stacked_masks_cat, alpha=alpha)
    elif folder_dropdown == "building":
        weights = calculate_building_distance_weights(
            stacked_masks, sigma=sigma, c=c, alpha=alpha
        )
    else:
        raise ValueError(
            "Invalid value for folder_dropdown. Choose from 'frequency', 'google', 'size', or 'building'."
        )

    return weights
