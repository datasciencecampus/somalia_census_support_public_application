#!/usr/bin/env python
# coding: utf-8
# %%

# %%


""" Script for compute class counts function """


# %%


import pandas as pd
import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from functions_library import setup_sub_dir


# %%


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


# %%


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate precision, recall, F1-score, and accuracy for each class.

    Args:
        y_true (numpy.ndarray): Array of true labels.
        y_pred (numpy.ndarray): Array of predicted labels.
        class_names (list): List of class names.

    Returns:
        pandas.DataFrame: DataFrame containing the calculated metrics for each class.

    """
    # Calculate the confusion matrix
    conf_mat = confusion_matrix(y_true.ravel(), y_pred.ravel())

    # Calculate the precision, recall, and F1-score for each class
    num_classes = conf_mat.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    accuracy = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = conf_mat[i, i]
        false_positives = np.sum(conf_mat[:, i]) - true_positives
        false_negatives = np.sum(conf_mat[i, :]) - true_positives

        precision[i] = true_positives / (true_positives + false_positives)
        recall[i] = true_positives / (true_positives + false_negatives)
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

        true_negatives = (
            np.sum(conf_mat)
            - np.sum(conf_mat[:, i])
            - np.sum(conf_mat[i, :])
            + true_positives
        )
        accuracy[i] = (true_positives + true_negatives) / np.sum(conf_mat)

    # Create the DataFrame
    metrics_df = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "Accuracy": accuracy,
        }
    )

    return metrics_df


def calculate_tile_metrics(y_pred, y_test_argmax, class_names, filenames):
    """
    Calculate performance metrics for each tile and create a DataFrame.

    Args:
        y_pred (np.ndarray): Array of predicted labels, where each element corresponds to a tile.
        y_test_argmax (np.ndarray): Array of true labels, where each element corresponds to a tile.
        class_names (list): List of class names.
        filenames (list): List of filesnames for each tile.

    Returns:
        pandas.DataFrame: DataFrame containing performance metrics for each tile.

    """
    metrics_df = pd.DataFrame()
    filenames_without_background = []

    for tile in range(len(y_test_argmax)):
        y_true = y_test_argmax[tile]
        y_single_pred = np.argmax(y_pred[tile], axis=-1)

        if not filenames[tile].endswith("background"):
            tile_metrics = calculate_metrics(y_true, y_single_pred, class_names)
            tile_metrics.index = tile_metrics.index + 1
            tile_metrics = tile_metrics.stack()
            tile_metrics.index = tile_metrics.index.map("{0[1]}_{0[0]}".format)
            tile_metrics.to_frame().T
            filenames_without_background.append(filenames[tile])
            metrics_df = pd.concat(
                [metrics_df, tile_metrics], axis=1, ignore_index=True
            )

    metrics_df = metrics_df.T
    metrics_df.insert(0, "tile", filenames_without_background)

    return metrics_df


def plot_confusion_matrix(y_true, y_pred):

    labels = ["background", "building", "tent"]

    conf_mat = confusion_matrix(y_true.ravel(), y_pred.ravel())

    # calculate the percentages
    row_sums = conf_mat.sum(axis=1)
    conf_mat_percent = conf_mat / row_sums[:, np.newaxis]

    display = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat_percent, display_labels=labels
    )

    # plot the confusion matrix
    display.plot(cmap="cividis", values_format=".2%")

    # show the plot
    plt.show()


def remove_rows_by_index(df, word):

    df = df.reset_index()
    mask = df["Tile"].str.endswith(word)
    df = df[~mask]
    df = df.set_index("Tile")

    return df


def compute_predicted_counts(y_pred, filenames_test):
    """
    Compute the counts of each class in each tile for predicted arrays.

    Args:
        y_pred (ndarray): Predicted array with shape (batch_size, height, width, num_classes).
        filenames_test (ndarray): Array with filenames for each tile.

    Returns:
        DataFrame: Pandas DataFrame containing the counts of each class in each sample.
    """

    class_counts_pred = []
    class_labels = {0: "Background", 1: "Building", 2: "Tent"}

    for tile_index in range(y_pred.shape[0]):
        tile_counts_pred = {}

        for class_index, class_label in class_labels.items():
            if class_label == "background":
                continue

            # Extract the predicted mask for the current class
            class_mask_pred = np.argmax(y_pred[tile_index], axis=-1) == class_index

            # Perform connected component analysis for predicted counts
            num_labels_pred, labeled_mask_pred = cv2.connectedComponents(
                class_mask_pred.astype(np.uint8)
            )
            # Count the number of objects for the current class in the current tile
            num_objects_pred = num_labels_pred - 1
            tile_counts_pred[class_label] = num_objects_pred

        class_counts_pred.append(tile_counts_pred)

    # Create a pandas DataFrame
    df = pd.DataFrame(columns=list(class_labels.values()))

    # Populate the DataFrame with the predicted counts for each class in each tile
    for tile_index in range(y_pred.shape[0]):
        tile_counts_pred = class_counts_pred[tile_index]
        row_data = {}

        for class_label in class_labels.values():
            if class_label != "Background":
                pred_count = tile_counts_pred.get(class_label, 0)
                row_data[class_label] = pred_count

        df = df.append(row_data, ignore_index=True)

    df.index = filenames_test
    df.index.name = "Tile"
    df = df.reindex(columns=["Tent", "Building"])

    # change column names for Tent and Building
    df = df.rename(columns={"Tent": "tent_computed", "Building": "building_computed"})

    return df


def compute_actual_counts(filenames_test):
    """
    Compute the counts of each class in each tile from the actual JSON files.

    Args:
        filenames_test (ndarray): Array with filenames for each tile.

    Returns:
        DataFrame: Pandas DataFrame containing the counts of each class in each sample.
    """

    class_counts_actual = []

    # Load in the actual feature numbers from geoJSONs
    features_file = mask_dir.joinpath("feature_dict.json")

    with open(features_file) as f:
        feature_data = json.load(f)

    for filename in filenames_test:
        if filename in feature_data:
            class_counts_actual.append(feature_data[filename])
        else:
            print(filename)

    # Create a pandas DataFrame
    df = pd.DataFrame(
        columns=[
            "Tile",
            "Tent_actual",
            "Building_actual",
            "tent_average",
            "building_average",
        ]
    )

    # Populate the DataFrame with the actual counts for each class in each tile
    for tile_index in range(len(filenames_test)):
        tile_counts_actual = class_counts_actual[tile_index]
        row_data = {"Tile": filenames_test[tile_index]}

        for class_label in ["Tent", "Building"]:
            actual_count = tile_counts_actual.get(class_label, 0)
            row_data[class_label + "_actual"] = actual_count

            row_data["tent_average"] = feature_data[filenames_test[tile_index]].get(
                "tent_average", 0
            )
            row_data["building_average"] = feature_data[filenames_test[tile_index]].get(
                "building_average", 0
            )

        df = df.append(row_data, ignore_index=True)
    df.set_index("Tile", inplace=True)

    df.rename(columns=lambda x: x.lower(), inplace=True)

    df = df.reindex(
        columns=["tent_actual", "building_actual", "tent_average", "building_average"]
    )

    return df


def compute_object_counts(
    y_pred, filenames_test, average_building_size, average_tent_size
):
    """
    Compute the number of individual objects for each class in a new object based on average sizes.

    Args:
        y_pred (ndarray): Predicted array with shape (batch_size, height, width, num_classes).
        filenames_test (ndarray): Array with filenames for each tile.
        average_building_size (float): Average size of a building object in square meters.
        average_tent_size (float): Average size of a tent object in square meters.

    Returns:
        DataFrame: Pandas DataFrame containing the number of individual objects for each class in each sample.
    """

    class_labels = {1: "building", 2: "tent"}
    object_counts = []

    for tile_index in range(y_pred.shape[0]):
        tile_objects = {}
        for class_index, class_label in class_labels.items():

            # Extract the predicted mask for the current class
            class_mask_pred = np.argmax(y_pred[tile_index], axis=-1) == class_index
            # Compute the sum of pixels for the current class in the current tile
            pixel_sum = np.sum(class_mask_pred)

            # Convert pixel sum to area in square meters
            area = pixel_sum * 0.5  # Assuming each pixel represents 0.5m
            if class_label == "building":
                object_count = area / average_building_size
            elif class_label == "tent":
                object_count = area / average_tent_size
            else:
                object_count = 0

            tile_objects[class_label + "_object_count"] = object_count
        object_counts.append(tile_objects)

    # Create a pandas DataFrame
    df = pd.DataFrame(
        columns=["Tile"]
        + [class_label + "_object_count" for class_label in class_labels.values()]
    )

    # Populate the DataFrame with the number of objects for each class in each tile
    for tile_index, filename in enumerate(filenames_test):
        tile_objects = object_counts[tile_index]
        row_data = {"Tile": filename}
        for class_label in class_labels.values():
            row_data[class_label + "_object_count"] = tile_objects.get(
                class_label + "_object_count", 0
            )

        df = df.append(row_data, ignore_index=True)

    df.set_index("Tile", inplace=True)

    return df


def compute_pixel_counts(y_pred, filenames_test):
    """
    Compute the number of individual objects for each class in a new object based on average sizes.

    Args:
        y_pred (ndarray): Predicted array with shape (batch_size, height, width, num_classes).
        filenames_test (ndarray): Array with filenames for each tile.

    Returns:
        DataFrame: Pandas DataFrame containing the number of individual objects for each class in each sample.
    """

    class_labels = {1: "building", 2: "tent"}
    object_counts = []

    pixel_area = 0.5

    for tile_index in range(y_pred.shape[0]):
        tile_objects = {}
        for class_index, class_label in class_labels.items():

            # Extract the predicted mask for the current class
            class_mask_pred = np.argmax(y_pred[tile_index], axis=-1) == class_index
            # Compute the sum of pixels for the current class in the current tile
            pixel_sum = np.sum(class_mask_pred)

            # Convert pixel sum to area in square meters
            area = pixel_sum * pixel_area  # Assuming each pixel represents 0.5m
            tile_objects[class_label + "_area"] = area

        object_counts.append(tile_objects)

    # Create a pandas DataFrame
    df = pd.DataFrame(
        columns=["Tile"]
        + [class_label + "_area" for class_label in class_labels.values()]
    )

    # Populate the DataFrame with the number of objects for each class in each tile
    for tile_index, filename in enumerate(filenames_test):
        tile_objects = object_counts[tile_index]
        row_data = {"Tile": filename}
        for class_label in class_labels.values():
            row_data[class_label + "_area"] = tile_objects.get(class_label + "_area", 0)

        df = df.append(row_data, ignore_index=True)

    df.set_index("Tile", inplace=True)

    return df


def make_pixel_stats(dataframe):
    """
    Calculates and merges statistics for building and tent object counts.

    Args:
        dataframe (DataFrame): Input DataFrame containing the columns 'Building_actual', 'Tent_actual',
                              'tent_average', 'building_average', Building_object_count', 'Tent_object_count'.

    Returns:
        DataFrame: Merged DataFrame containing statistics for calculated columns and reference columns

    """
    # reset index to have 'tile' as regular column
    dataframe_reset = dataframe.reset_index()

    # columns for ref
    col_for_ref = ["building_actual", "tent_actual", "tent_average", "building_average"]

    # take reference figures and group by tile for first df
    ref_fig_df = dataframe_reset.groupby("Tile")[col_for_ref].max()

    # columns for stats
    col_for_stats = ["building_object_count", "tent_object_count"]

    # second df with summary stats for selected columns
    pixel_stats_df = dataframe_reset.groupby("Tile")[col_for_stats].agg(
        ["min", "max", "mean"]
    )
    # rename columns with building and tent to keep on one level
    pixel_stats_df.columns = [f"{col}_{label}" for col, label in pixel_stats_df.columns]

    # merge two df on 'tile' column
    pixel_stats_final_df = pixel_stats_df.merge(ref_fig_df, on="Tile")

    return pixel_stats_final_df


def make_computed_stats(dataframe):
    """
    Calculates and merges statistics for building and tent object counts.

    Args:
        dataframe (DataFrame): Input DataFrame containing the columns 'Building_actual', 'Tent_actual',
                              'tent_average', 'building_average', Building_object_count', 'Tent_object_count'.

    Returns:
        DataFrame: Merged DataFrame containing statistics for calculated columns and reference columns

    """
    # reset index to have 'tile' as regular column
    dataframe_reset = dataframe.reset_index()

    # columns for ref
    col_for_ref = ["building_actual", "tent_actual"]

    # take reference figures and group by tile for first df
    ref_fig_df = dataframe_reset.groupby("Tile")[col_for_ref].max()

    # columns for stats
    col_for_stats = ["building_computed", "tent_computed"]

    # second df with summary stats for selected columns
    stats_df = dataframe_reset.groupby("Tile")[col_for_stats].agg(
        ["min", "max", "mean"]
    )
    # rename columns with building and tent to keep on one level
    stats_df.columns = [f"{col}_{label}" for col, label in stats_df.columns]

    # merge two df on 'tile' column
    stats_final_df = stats_df.merge(ref_fig_df, on="Tile")

    return stats_final_df


# %%
