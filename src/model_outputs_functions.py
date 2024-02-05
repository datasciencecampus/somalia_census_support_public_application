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
from rasterio import features
from shapely.geometry import Polygon
import geopandas as gpd
from IPython.display import display


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


def plot_confusion_matrix(y_true, y_pred, labels):

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
            "tent_actual",
            "building_actual",
            "tent_average",
            "building_average",
        ]
    )
    # Populate the DataFrame with the actual counts for each class in each tile
    for tile_index in range(len(filenames_test)):
        tile_counts_actual = class_counts_actual[tile_index]
        row_data = {"Tile": filenames_test[tile_index]}

        for class_label in ["tent", "building"]:
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
            tile_objects[class_label + "_pixel_sum"] = pixel_sum

        object_counts.append(tile_objects)

    # Create a pandas DataFrame
    df = pd.DataFrame(
        columns=["Tile"]
        + [class_label + "_area" for class_label in class_labels.values()]
        + [class_label + "_pixel_sum" for class_label in class_labels.values()]
    )

    # Populate the DataFrame with the number of objects for each class in each tile
    for tile_index, filename in enumerate(filenames_test):
        tile_objects = object_counts[tile_index]
        row_data = {"Tile": filename}
        for class_label in class_labels.values():
            row_data[class_label + "_area"] = tile_objects.get(class_label + "_area", 0)
            row_data[class_label + "_pixel_sum"] = tile_objects.get(
                class_label + "_pixel_sum", 0
            )

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


def create_grouped_filenames(filenames):
    """
    Create a DataFrame grouped by 'tile_name' with a 'index_number' column that
    contains comma-separated index positions and a 'count' column of how many times
    in the training dataset.

    Parameters:
    - filenames (npy.ndarray): 1D NumPy array containing 'tile_name' values.

    Returns:
    - pd.DataFrame: DataFrame with columns 'tile_name', 'index_number', and 'count'.
    """
    # reshape the 1D array to a 2D array with a single column
    filenames_reshaped = filenames.reshape(-1, 1)

    # create a DataFrame with 'tile_name' column
    filenames_df = pd.DataFrame(filenames_reshaped, columns=["tile_name"])

    # group by 'tile_name' and aggregate the index positions into a comma-separated string
    grouped_df = (
        filenames_df.groupby("tile_name")
        .apply(lambda x: ",".join(map(str, x.index.tolist())))
        .reset_index(name="index_number")
    )

    # add a 'count' column containing the count of index positions within each group
    grouped_df["count"] = grouped_df["index_number"].apply(lambda x: len(x.split(",")))

    return grouped_df


def generate_class_names(n_classes):
    """
    Generate class names based on the number of classes.

    Parameters:
    - n_classes (int): Number of classes.

    Returns:
    - list: List of class names.
    """
    if n_classes == 3:
        class_names = ["background", "building", "tent"]
    elif n_classes == 5:
        class_names = [
            "background",
            "building",
            "tent",
            "building_border",
            "tent_border",
        ]
    else:
        raise ValueError("Unsupported number of classes. Please provide either 3 or 5.")

    return class_names


# metrics function to filter on widget
def update_displayed_data(tile_metrics_df, selected_tile):
    selected_tile_data = tile_metrics_df[tile_metrics_df.index == selected_tile]

    if not selected_tile_data.empty:
        display(selected_tile_data)


def plot_images_for_tile(model, X_test, y_test_argmax, grouped_tiles_df, selected_tile):
    """
    Plot images for a specific tile based on the provided DataFrame and selected tile name.

    Parameters:
    - grouped_tiles_df (pandas.DataFrame): DataFrame containing information about grouped tiles.
    - selected_tile (str): Name of the selected tile.

    Returns:
    list: List of matplotlib figures displaying images for the selected tile.
    """
    specific_tile_df = grouped_tiles_df[grouped_tiles_df["tile_name"] == selected_tile]

    num_count = specific_tile_df["count"].iloc[0]
    index_numbers = specific_tile_df["index_number"].iloc[0].split(",")

    if num_count > 0 and index_numbers:
        plots = []

        for i in range(min(num_count, len(index_numbers))):
            test_img_number = int(index_numbers[i])

            test_img = X_test[test_img_number]
            ground_truth = y_test_argmax[test_img_number]
            test_img_input = np.expand_dims(test_img, 0)
            prediction = model.predict(test_img_input)
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]

            test_img = test_img[:, :, :3]
            test_img = test_img[:, :, ::-1]

            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
            axs[0].imshow(test_img)
            axs[0].set_title("Testing Image")

            axs[1].imshow(ground_truth)
            axs[1].set_title("Testing Label")

            axs[2].imshow(predicted_img)
            axs[2].set_title("Prediction on test image")

            plots.append(fig)

        return plots
    else:
        return []


def create_mask_dict(predicted_img, unique_classes):
    """
    Create a dictionary of masks based on predicted image and unique classes.

    Parameters:
    - predicted_img (npy.ndarray): Predicted image with class labels.
    - unique_classes (list): List of unique class labels.

    Returns:
    dict: Dictionary containing masks for each unique class.
    """
    return {
        f"mask_{cls}": (predicted_img == cls).astype(np.uint8) for cls in unique_classes
    }


def extract_shapes_from_masks(mask_dict):
    """
    Extract shapes from a dictionary of masks.

    Parameters:
    - mask_dict (dict): Dictionary containing masks.

    Returns:
    dict: Dictionary containing shapes extracted from masks.
    """
    return {
        title: features.shapes(mask, mask=mask, connectivity=4)
        for title, mask in mask_dict.items()
    }


def shapes_to_geopandas(shapes, category, filename, index_number):
    """
    Convert shapes to GDF.

    Parameters:
    - shapes (dict): Dictionary containing shapes information.
    - category (str): Category label for the GeoDataFrame.
    - filename (str): Filename associated with the shapes.

    Returns:
    list: List of dictionaries containing geometry, type, and filename for each shape.
    """
    return [
        {
            "geometry": Polygon(shape["coordinates"][0]),
            "type": category,
            "filename": filename,
            "index_num": index_number,
        }
        for shape, _ in shapes
        if shape["type"] == "Polygon"
    ]


def process_tile(model, tile, unique_classes, filename, index_number):
    """
    Process a single tile by generating masks, extracting shapes, and creating GDF.

    Parameters:
    - test_img (npy.ndarray): Input image tile.
    - unique_classes (list): List of unique class labels.
    - filename (str): Filename associated with the tile.

    Returns:
    geopandas.GeoDataFrame: Combined GeoDataFrame containing buildings and tents information.
    """
    test_img_input = np.expand_dims(tile, 0)
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    # Create mask dictionary
    mask_dict = create_mask_dict(predicted_img, unique_classes)

    # Extract shapes from masks
    shapes_dict = extract_shapes_from_masks(mask_dict)

    # Extract shapes for buildings and tents
    buildings_shapes = shapes_dict.get("mask_1", None)
    tents_shapes = shapes_dict.get("mask_2", None)

    # Convert shapes to GDF
    buildings_geometries = shapes_to_geopandas(
        buildings_shapes, "buildings", filename, index_number
    )
    tents_geometries = shapes_to_geopandas(
        tents_shapes, "tents", filename, index_number
    )

    # Create GDF for buildings and tents
    buildings_gdf = gpd.GeoDataFrame(buildings_geometries)
    tents_gdf = gpd.GeoDataFrame(tents_geometries)

    # Concatenate GDF
    combined_gdf = gpd.GeoDataFrame(
        pd.concat([buildings_gdf, tents_gdf], ignore_index=True)
    )

    return combined_gdf


def process_json_files(json_dir, grouped_counts):
    """
    Process JSON files in a directory and return a DataFrame with selected columns.

    Parameters:
    - json_dir (Path): Path object pointing to the directory containing JSON files.

    Returns:
    - pd.DataFrame: DataFrame containing selected columns from the JSON files.
    """

    # List all JSON files in the directory
    json_files = list(json_dir.glob("*.json"))

    # Load data from each JSON file and store it in a list
    all_feature_data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            feature_data = json.load(f)
            all_feature_data.append(feature_data)

    # Concatenate all feature data into a DataFrame
    feature_data_df = pd.concat(
        [
            pd.DataFrame(data).T.assign(filename=json_file.stem)
            for json_file, data in zip(json_files, all_feature_data)
        ]
    )

    # Select and reset index for specified columns
    selected_columns_df = (
        feature_data_df[["building", "tent"]]
        .reset_index()
        .rename(columns={"index": "filename"})
    )

    # Rename columns
    renamed_columns_df = selected_columns_df[["filename", "building", "tent"]].rename(
        columns={"building": "building_actual", "tent": "tent_actual"}
    )

    # Merge with existing DataFrame
    building_polygon_counts = pd.merge(
        grouped_counts,
        renamed_columns_df[["filename", "building_actual", "tent_actual"]],
        on="filename",
        how="left",
    )

    return building_polygon_counts


def building_stats(building_polygon_counts):
    """
    Preprocess building counts DataFrame by filling NaN values, setting numerical columns to whole numbers,
    and creating columns to show count and percentage differences between actual and computed polygons.

    Parameters:
    - building_polygon_counts (pd.DataFrame): DataFrame containing building counts data.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame with the specified modifications.
    """

    # Fill NaN values in 'building_actual' and 'tent_actual' columns with zeros
    building_polygon_counts["building_actual"].fillna(0, inplace=True)
    building_polygon_counts["tent_actual"].fillna(0, inplace=True)

    # Set specified numerical columns to whole numbers
    numerical_columns = ["building_actual", "tent_actual"]
    building_polygon_counts[numerical_columns] = building_polygon_counts[
        numerical_columns
    ].astype(int)

    # Create columns to show count difference between actual and computed polygons
    building_polygon_counts["building_diff"] = (
        building_polygon_counts["building_actual"]
        - building_polygon_counts["buildings"]
    )
    building_polygon_counts["tent_diff"] = (
        building_polygon_counts["tent_actual"] - building_polygon_counts["tents"]
    )

    # Create columns to show percentage difference between actual and computed polygons
    building_polygon_counts["%_change_tent"] = (
        (building_polygon_counts["tent_diff"] / building_polygon_counts["tents"]) * 100
    ).round(0)
    building_polygon_counts["%_change_building"] = (
        (
            building_polygon_counts["building_diff"]
            / building_polygon_counts["buildings"]
        )
        * 100
    ).round(0)

    return building_polygon_counts
