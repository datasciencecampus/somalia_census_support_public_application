#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" Script for compute class counts function """


# In[ ]:


import pandas as pd
import cv2
import json
import numpy as np
from pathlib import Path
from functions_library import setup_sub_dir


# In[ ]:


# set data directory
data_dir = Path.cwd().parent.joinpath("data")

# set-up model directory for model and outputs
models_dir = setup_sub_dir(Path.cwd().parent, "models")
outputs_dir = setup_sub_dir(Path.cwd().parent, "outputs")

# set training_data directory within data folder
training_data_dir = data_dir.joinpath("training_data")

# set img and mask directories within training_data directory
img_dir = training_data_dir.joinpath("img")
mask_dir = training_data_dir.joinpath("mask")


# In[ ]:


def compute_class_counts(y_pred, y_test, filenames_test):
    """
    Compute the counts of each class in each tile for predicted and test arrays.

    Args:
        y_pred (ndarray): Predicted array with shape (batch_size, height, width, num_classes).
        y_test (ndarray): Actual array with shape (batch_size, height, width, num_classes).
        filenames_test(ndarrary): Array with... (batch_size) # not sure what to put here


    Returns:
        DataFrame: Pandas DataFrame containing the counts of each class in each sample.

    """

    class_counts_pred = []
    class_counts_actual = []

    class_labels = {0: "Background", 1: "Building", 2: "Tent"}
    
    # Load in the actual feature numbers from geoJSONs
    features_file = mask_dir.joinpath("feature_dict.json")
    with open(features_file) as f:
        feature_data = json.load(f)
    
    # Counts predicted arrays using connected Connected Components

    for tile_index in range(y_pred.shape[0]):
        tile_counts_pred = {}
        tile_counts_actual = {}

        for class_index, class_label in class_labels.items():
            if class_label == "Background":
                continue

            # Extract the predicted mask for the current class
            class_mask_pred = np.argmax(y_pred[tile_index], axis=-1) == class_index

            # Perform connected component analysis for predicted counts
            num_labels_pred, labeled_mask_pred = cv2.connectedComponents(
                class_mask_pred.astype(np.uint8)
            )

            # Count the number of objects for the current class in the current tile (excluding background label)
            num_objects_pred = num_labels_pred - 1

            tile_counts_pred[class_label] = num_objects_pred

        class_counts_pred.append(tile_counts_pred)
    
    
    for filename in filenames_test:
            if filename in feature_data:
                class_counts_actual.append(feature_data[filename])
            else:
                print(filename)

    # Create a pandas DF
    df = pd.DataFrame(columns=["Tile"] + list(class_labels.values()))
  
    # Populate the DataFrame with the actual counts for each class in each tile

    for tile_index in range(y_pred.shape[0]):
        tile_counts_pred = class_counts_pred[tile_index]
        tile_counts_actual = class_counts_actual[tile_index]
        row_data = {"Tile": tile_index}

        for class_label in class_labels.values():
            if class_label != "Background":
                pred_count = tile_counts_pred.get(class_label, 0)
                actual_count = tile_counts_actual.get(class_label, 0)
                row_data[class_label] = pred_count
                row_data[class_label + "_actual"] = actual_count

        df = df.append(row_data, ignore_index=True)

    # probably want to change this when file name added
    df["Tile"] = filenames_test
    df = df.reindex(columns=['Tile', 'Tent', 'Tent_actual', 'Building', 'Building_actual'])
    return df
