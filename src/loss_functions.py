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


def cross_entropy(y_true, y_pred):
    epsilon = 1e-07  # Small constant to avoid division by zero
    y_pred = tf.clip_by_value(
        y_pred, epsilon, 1 - epsilon
    )  # Clip values to avoid log(0)
    loss = -tf.reduce_mean(y_true * tf.math.log(y_pred), axis=-1)
    return loss


def weighted_multi_class_loss(
    y_test, y_pred, class_weights, weights_ce=1.0, weights_dice=1.0, weights_focal=1.0
):
    """
    Calculate weighted multi-class loss.

    This function computes a weighted multi-class loss by combining cross-entropy,
    dice, and focal losses with respective weights.

    Args:
        y_test (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.
        class_weights (list): Weights for each class.
        weights_ce (float): Weight for cross-entropy loss.
        weights_dice (float): Weight for dice loss.
        weights_focal (float): Weight for focal loss.

    Returns:
        tensor: Weighted multi-class loss.

    Example:
        loss = weighted_multi_class_loss(y_true, y_pred, [0.7, 0.15, 0.15], 1.0, 1.0, 1.0)
    """
    # cross entropy loss
    loss_ce = cross_entropy(y_test, y_pred)

    # dice loss
    loss_dice = dice_loss(y_test, y_pred)

    # focal loss
    loss_focal = focal_loss(y_test, y_pred)

    # apply class weights to each loss
    weighted_losses = (
        [class_weights[i] * loss_ce for i in range(len(class_weights))]
        + [class_weights[i] * loss_dice for i in range(len(class_weights))]
        + [class_weights[i] * loss_focal for i in range(len(class_weights))]
    )

    # combine weighted losses with respective weights
    loss = (
        weights_ce * weighted_losses[0]
        + weights_dice * weighted_losses[1]
        + weights_focal * weighted_losses[2]
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


def get_combined_loss(y_true, y_pred):
    loss = [focal_loss, dice_loss]
    return loss


def get_loss_function(loss_dropdown, class_weights=None):
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

    elif loss_dropdown == "segmentation_models":
        loss = get_sm_loss(class_weights)

    elif loss_dropdown == "focal_tversky":
        loss = focal_tversky_loss

    elif loss_dropdown == "tversky":
        loss = tversky_loss

    return loss
