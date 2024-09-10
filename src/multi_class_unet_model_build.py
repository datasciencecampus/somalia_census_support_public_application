"""
Multi-class U-Net model template courtesy of GitHub user bnsreenu.
Original code:
https://github.com/bnsreenu/python_for_microscopists/blob/master/228_semantic_segmentation_of_aerial_imagery_using_unet/simple_multi_unet_model.py
and associated YouTube video:
https://youtu.be/jvZm8REF2KY

----
Standard UNET
Model not compiled here, instead will be done externally to make it
easy to test various loss functions and optimizers.

"""
from keras import backend as K
from keras.layers import (  # BatchNormalization,; Lambda,; UpSampling2D,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from keras.models import Model
from sklearn.model_selection import train_test_split


def jacard_coef(y_test, y_pred):
    y_test_f = K.flatten(y_test)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_test_f * y_pred_f)
    return (intersection + 1.0) / (
        K.sum(y_test_f) + K.sum(y_pred_f) - intersection + 1.0
    )


def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    # NOTE: Compile the model in the main program to make it easy to test with various loss functions
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    return model


def split_data(
    all_stacked_images,
    stacked_masks_cat,
    all_stacked_filenames=None,
    validation_images=None,
    validation_masks_cat=None,
    validation_filenames=None,
):
    if validation_images is not None:

        X_train = all_stacked_images
        y_train = stacked_masks_cat
        filename_train = all_stacked_filenames

        X_test = validation_images
        y_test = validation_masks_cat
        filename_test = validation_filenames
    else:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            filename_train,
            filename_test,
        ) = train_test_split(
            all_stacked_images,
            stacked_masks_cat,
            all_stacked_filenames,
            test_size=0.20,
            random_state=42,
        )
    return X_train, X_test, y_train, y_test, filename_train, filename_test


def get_model(n_classes, img_height, img_width, num_channels):
    return multi_unet_model(
        n_classes=n_classes,
        IMG_HEIGHT=img_height,
        IMG_WIDTH=img_width,
        IMG_CHANNELS=num_channels,
    )
