# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# import standard and third party libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### Set-up filepaths

# %%
data_dir = Path.cwd().parent.joinpath("data")

# %% [markdown]
# ### Load raster arrays

# %%
with open(data_dir.joinpath('normalised_sat_raster.npy'), 'rb') as f:
    normalised_sat_raster = np.load(f)

# %%
with open(data_dir.joinpath('training_mask_raster.npy'), 'rb') as f:
    training_mask_raster = np.load(f)

# %%
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(training_mask_raster)
plt.subplot(122)
plt.imshow(normalised_sat_raster[:,:,:3])
plt.show()

# %% [markdown]
# ### Process training parameters

# %%
num_channels, img_height, img_width = normalised_sat_raster.shape

# %%
# keras installation requires tensoreflow installed in back
from multi_class_unet_model_build import multi_unet_model, jacard_coef

# %%
training_mask_raster.shape

# %%
#Parameters for model
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(
    'balanced',
    np.unique(training_mask_raster),
    np.ravel(training_mask_raster, order='C')
)
print(weights)

# %%
n_classes = len(np.unique(training_mask_raster))
n_classes

# %%
from keras.utils import to_categorical

# one-hot encode building classes in training mask
labels_categorical = to_categorical(training_mask_raster, num_classes=n_classes)
labels_categorical.shape

# %%
import keras


# %%
import segmentation_models as sm

# create custom loss function for model training
dice_loss = sm.losses.DiceLoss(class_weights=weights) # corrects for class imbalance
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

# %%
labels_categorical = np.repeat(labels_categorical[...,None], 7, axis = 3)
labels_categorical.shape

# %%
labels_categorical = np.transpose(labels_categorical, axes = [3, 0, 1, 2])
labels_categorical.shape

# %%
#TODO: for each training tile, generate additional ones by rotating them and mirroring them
# - will need to do same for corresponding masks of course

#NOTE: These have to be order with list index, then x, y, band/class
stacked_training_rasters = np.repeat(normalised_sat_raster[...,None], 7, axis = 3)
stacked_training_rasters = np.transpose(stacked_training_rasters, axes = [3, 0, 1, 2])

stacked_training_rasters.shape

# %%
from sklearn.model_selection import train_test_split

# stacked_training_rasters = stack of individual training raster tiles of same dimensions

X_train, X_test, y_train, y_test = train_test_split(
    stacked_training_rasters,
    labels_categorical,
    test_size = 0.20,
    random_state = 42
    )

# %%
X_train.shape


# %%
def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=img_height, IMG_WIDTH=img_width, IMG_CHANNELS=num_channels)

model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()


history1 = model.fit(X_train,
                     y_train,
                     batch_size = 16,
                     verbose=1,
                     epochs=100,
                     validation_data=(X_test, y_test),
                     shuffle=False
                    )
