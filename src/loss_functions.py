""" Script for loss functions """

import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow.keras import backend as K
import tensorflow as tf
import segmentation_models as sm


def dice_loss(y_test, y_pred):
    """
    Calculate Dice loss.

    This function computes the Dice loss, which is a measure of overlap between
    the ground truth and predicted segmentation masks.

    Args:
        y_test (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.

    Returns:
        tensor: Dice loss.

    Example:
        loss = dice_loss(y_true, y_pred)
    """
    smooth = 1e-5
    intersection = K.sum(y_test * y_pred)
    union = K.sum(y_test) + K.sum(y_pred)
    dice_coef = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice_coef


def focal_loss(y_test, y_pred, gamma=2.0, alpha=0.25):
    """
    Calculate Focal loss.

    This function computes the Focal loss, which is designed to address class
    imbalance in binary classification tasks by down-weighting easy examples
    and focusing on hard examples.

    Args:
        y_test (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.
        gamma (float): Focusing parameter (default is 2.0).
        alpha (float): Balancing parameter (default is 0.25).

    Returns:
        tensor: Focal loss.

    Example:
        loss = focal_loss(y_true, y_pred)
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    cross_entropy = -y_test * K.log(y_pred)
    weight = alpha * y_test * K.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy

    return K.mean(loss)


def weighted_multi_class_loss(
    y_test,
    y_pred,
    weights_distance,
    weights_size,
    weights_ce,
    weights_dice,
    weights_focal,
):
    """
    Calculate weighted multi-class loss.

    This function computes a weighted multi-class loss by combining cross-entropy,
    dice, and focal losses with respective weights.

    Args:
        y_test (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.
        weights_distance (float): Weight for distance loss.
        weights_size (float): Weight for size loss.
        weights_ce (float): Weight for cross-entropy loss.
        weights_dice (float): Weight for dice loss.
        weights_focal (float): Weight for focal loss.

    Returns:
        tensor: Weighted multi-class loss.

    Example:
        loss = weighted_multi_class_loss(y_true, y_pred, 1.0, 0.8, 1.0, 1.0, 1.0)
    """
    # cross entropy loss
    loss_ce = K.sparse_categorical_crossentropy(y_test, y_pred)

    # dice loss
    loss_dice = dice_loss(y_test, y_pred)

    # focal loss
    loss_focal = focal_loss(y_test, y_pred)

    # calculate pixel weights
    weights = weights_distance * weights_size

    # combine losses with weights
    loss = weights * (
        weights_ce * loss_ce + weights_dice * loss_dice + weights_focal * loss_focal
    )

    return loss


def focal_tversky_loss(y_test, y_pred, alpha=0.7, beta=0.3, gamma=1.0, smooth=1e-06):
    """
    Focal Tversky loss

    Args:
        y_test (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.
        alpha (float, optional): Weight of false negatives. Defaults to 0.7.
        beta (float, optional): Weight of false positives. Defaults to 0.3.
        gamma (float, optional): Focusing parameter. Defaults to 1.0.
        smooth (float, optional): Smoothing term to avoid division by zero. Defaults to 1e-6.

    Returns:
        tensor: The Focal Tversky loss.

    """

    y_test_pos = tf.keras.backend.flatten(y_test)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    test_pos = tf.reduce_sum(y_test_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_test_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_test_pos) * y_pred_pos)

    tversky_coef = (test_pos + smooth) / (
        test_pos + alpha * false_neg + beta * false_pos + smooth
    )
    focal_tversky = tf.pow((1 - tversky_coef), gamma)
    loss = focal_tversky

    return loss


def get_custom_loss(weights_distance, weights_size):
    """

    This function returns a lambda function that calculates a custom loss using
    weighted multi-class loss function with provided weights.

    Args:
        weights_distance (float): Weight for distance loss.
        weights_size (float): Weight for size loss.

    Returns:
        function: Lambda function for custom loss calculation.

    Example:
        custom_loss_func = get_custom_loss(0.8, 0.5)
        loss = custom_loss_func(y_true, y_pred)
    """
    weights_ce = 1
    weights_dice = (
        1  # if accuracy low trying increasing to place more emphasis on Dice loss
    )
    weights_focal = 1  # helps deal with imbalanced data. Range from 0.5 to 5
    return lambda y_test, y_pred: weighted_multi_class_loss(
        y_test,
        y_pred,
        weights_distance,
        weights_size,
        weights_ce,
        weights_dice,
        weights_focal,
    )


def get_sm_loss(class_weights):
    return sm.losses.DiceLoss(class_weights) + sm.losses.CategoricalFocalLoss()


def tversky_loss(y_true, y_pred):
    """
    Compute the Tversky loss.

    Tversky loss is a measure of dissimilarity between two sets.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.

    Returns:
        tensor: The computed Tversky loss.

    """
    alpha = 0.7
    smooth = 1.0
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    tversky_coef = (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )
    return 1 - tversky_coef


def get_loss_function(
    loss_dropdown,
    weights=None,
    weights_distance=1,
    weights_size=1,
    weights_ce=0,
    weights_dice=0.7,
    weights_focal=0.3,
):
    """
    Get the selected loss function.

    Args:
        loss_dropdown (str): Name of the selected loss function.
        weights (tuple, optional): Weights for custom loss functions.
        weights_distance (float, optional): Weight for distance loss in custom loss functions.
        weights_size (float, optional): Weight for size loss in custom loss functions.
        weights_ce (float, optional): Weight for cross-entropy loss in custom loss functions.
        weights_dice (float, optional): Weight for dice loss in custom loss functions.
        weights_focal (float, optional): Weight for focal loss in custom loss functions.

    Returns:
        function or list of functions: The selected loss function(s).

    """
    if loss_dropdown == "dice":
        loss = dice_loss

    elif loss_dropdown == "focal":
        loss = focal_loss

    elif loss_dropdown == "combined":
        loss = [focal_loss, dice_loss]
        loss_weights = (weights_dice, weights_focal)

    elif loss_dropdown == "segmentation_models":
        loss = get_sm_loss(weights)

    elif loss_dropdown == "custom":
        loss = get_custom_loss(weights_distance, weights_size)

    elif loss_dropdown == "focal_tversky":
        loss = focal_tversky_loss

    elif loss_dropdown == "tversky":
        loss = tversky_loss

    elif loss_dropdown == "weighted_multi_class":
        loss = weighted_multi_class_loss
        loss_weights = (
            weights_distance,
            weights_size,
            weights_ce,
            weights_dice,
            weights_focal,
        )

    return loss, loss_weights
