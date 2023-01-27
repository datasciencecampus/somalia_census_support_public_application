# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Load packages
#
# *NOTE:* this notebook require Keras a installation, which itself requires tensorflow installed first. Installing tensorflow requires additional steps beyond a simple pip install. See https://www.tensorflow.org/install

# %%
# import standard and third party libraries
from pathlib import Path
import numpy as np
from multi_class_unet_model_build import multi_unet_model, jacard_coef
from sklearn.utils.class_weight import compute_class_weight
import segmentation_models as sm
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# %%
# import custom functions
from functions_library import (
    setup_sub_dir
)

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
normalised_sat_raster_uncropped = normalised_sat_raster

# %%
with open(data_dir.joinpath('training_mask_raster.npy'), 'rb') as f:
    training_mask_raster = np.load(f)

# %% [markdown]
# # Crop raster and masks
#
# UNET models downsample by a factor of repeatedly. So ideally want to work with tiles that are divible by two many times.
#
# Our training data is created in 600x600 pixel areas, so currently cropping rasters to tiles of sizes 576x576, since 576=9x(2^6).

# %%
img_size = 576

# %%
normalised_sat_raster = normalised_sat_raster[0:img_size, 0:img_size, :]
normalised_sat_raster.shape

# %%
training_mask_raster = training_mask_raster[0:img_size, 0:img_size]
training_mask_raster.shape

# %% [markdown]
# ## Process training parameters

# %%
img_height, img_width, num_channels = normalised_sat_raster.shape

# %% [markdown]
# ### Weights
#
# The reltaive weights between building classes is one parameter that can be tweaked and optimised.

# %%
#Parameters for model
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss

# Calaculates the relative frequency of each class within the lablled mask.
weights = compute_class_weight(
    'balanced',
    classes=np.unique(training_mask_raster),
    y=np.ravel(training_mask_raster, order='C')
)

# Alternatively, could try balanced weights between classes:
#weights = [0.25, 0.25, 0.25, 0.25]

print(weights)


# %% [markdown]
# ### Loss function
# The loss function is an additional parameter that can be tweaked and optimised.

# %%
# create custom loss function for model training
# this is inspired from the tutorial used to create this initial code
# TODO: Explore alternatives
dice_loss = sm.losses.DiceLoss(class_weights=weights) # corrects for class imbalance
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss) 

# %% [markdown]
# ### Metrics
# The metrics used to measure the model performance can be optimised also.

# %%
metrics = ['accuracy', jacard_coef]

# %% [markdown]
# ## Get data into format the model expects

# %%
n_classes = len(np.unique(training_mask_raster))
n_classes

# %%
# TODO: add procedure to transform each training tile and its mask.
# that is, created rotated and mirrored versions and stack them

# %%
# one-hot encode building classes in training mask
labels_categorical = to_categorical(training_mask_raster, num_classes=n_classes)

# duplicates the single training mask to simulate the stack of training data
# that will exist at some stage
#TODO: Remove this later!
labels_categorical = np.repeat(labels_categorical[...,None], 5, axis = 3)

# reorder the array to image list index, height, width, categorical class 
labels_categorical = np.transpose(labels_categorical, axes = [3, 0, 1, 2])

labels_categorical.shape

# %%
# duplicates the single training image to simulate the stack of training data
# that will exist at some stage
#TODO: Remove this later!
stacked_training_rasters = np.repeat(normalised_sat_raster[...,None], 5, axis = 3)

# reorder the array to image list index, height, width, categorical class 
stacked_training_rasters = np.transpose(stacked_training_rasters, axes = [3, 0, 1, 2])

stacked_training_rasters.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(
    stacked_training_rasters,
    labels_categorical,
    test_size = 0.20,
    random_state = 42
    )


# %%
def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=img_height, IMG_WIDTH=img_width, IMG_CHANNELS=num_channels)


# %%
model = get_model()

model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

model.summary()

num_epochs = 20

history1 = model.fit(X_train,
                     y_train,
                     batch_size = 16,
                     verbose=1,
                     epochs=num_epochs,
                     validation_data=(X_test, y_test),
                     shuffle=False
                    )

# %%
models_dir = setup_sub_dir(Path.cwd().parent, "models")
model.save(models_dir.joinpath(f'trail_run_{num_epochs}epochs_{img_size}pix_doolow.hdf5'))

# %%
y_pred = model.predict(X_test)

predicted_img=np.argmax(y_pred, axis=3)[0,:,:]

# %% [markdown]
# ## Temp: output  predicted image for checking visually
# matplotlib wont work in this environment currently, so need to switch environment and use the `model_results_exploration_notebook`.

# %%
with open(data_dir.joinpath('pred.npy'), 'wb') as f:
    np.save(f, y_pred)

# %%
