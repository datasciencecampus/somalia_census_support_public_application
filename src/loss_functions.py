""" Script for loss functions """

import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow.keras import backend as K
import numpy as np
import segmentation_models as sm


def dice_loss(y_true, y_pred):
    """Dice loss function."""
    smooth = 1e-5
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice_coef = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice_coef


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss function"""
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy

    return K.mean(loss)


def weighted_multi_class_loss(
    y_true,
    y_pred,
    weights_distance,
    weights_size,
    weights_ce,
    weights_dice,
    weights_focal,
):
    """weighted multi-class loss function"""

    # cross entropy loss
    loss_ce = K.sparse_categorical_crossentropy(y_true, y_pred)

    # dice loss
    loss_dice = dice_loss(y_true, y_pred)

    # focal loss
    loss_focal = focal_loss(y_true, y_pred)

    # calculate pixel weights
    weights = weights_distance * weights_size

    # combine losses with weights
    loss = weights * (
        weights_ce * loss_ce + weights_dice * loss_dice + weights_focal * loss_focal
    )

    return loss


def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.0, smooth=1e-06):
    """
    Focal Tversky loss

    Args:
        y_true (array-like): The ground truth segmentation mask.
        y_pred (array-like): The predicted segmentation mask.
        alpha (float, optional): Weight of false negatives. Defaults to 0.7.
        beta (float, optional): Weight of false positives. Defaults to 0.3.
        gamma (float, optional): Focusing paramter. Defaults to 1.0.
        smooth (float, optional): Smoothing term to avoid division by zero. Defaults to 1e-6.

    Returns:
        array-like: The Focal Tversky loss.

    """

    y_true_pos = np.ndarray.flatten(y_true)
    y_pred_pos = np.ndarray.flatten(y_pred)
    true_pos = np.sum(y_true_pos * y_pred_pos)
    false_neg = np.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = np.sum((1 - y_true_pos) * y_pred_pos)

    tversky_coef = (true_pos + smooth) / (
        true_pos + alpha * false_neg + beta * false_pos + smooth
    )
    focal_tversky = np.power((1 - tversky_coef), gamma)
    loss = focal_tversky

    return loss


def get_custom_loss(weights_distance, weights_size):
    weights_ce = 1
    weights_dice = (
        1  # if accuracy low trying increasing to place more emphasis on Dice loss
    )
    weights_focal = 1  # helps deal with imbalanced data. Range from 0.5 to 5
    return lambda y_true, y_pred: weighted_multi_class_loss(
        y_true,
        y_pred,
        weights_distance,
        weights_size,
        weights_ce,
        weights_dice,
        weights_focal,
    )


def get_sm_loss(class_weights):
    return sm.losses.DiceLoss(class_weights) + sm.losses.CategoricalFocalLoss()


def get_combined_loss():
    loss = [focal_loss, dice_loss]
    loss_weights = [0.5, 0.5]
    return (loss, loss_weights)


def get_focal_tversky_loss():
    loss = focal_tversky_loss
    return loss
