""" Script for loss functions """


from tensorflow.keras import backend as K


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
