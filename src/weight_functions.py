""" Script for weights functions """

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label


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
