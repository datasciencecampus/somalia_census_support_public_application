""" Script of functions related to model training preprocessing. """


from pathlib import Path

import geopandas as gpd

import pandas as pd
import numpy as np
import rasterio as rio
from rasterio import features
import json


def process_geojson_file(mask_path):
    """
    Process a GeoJSON file.

    This function loads a GeoJSON file into a GeoPandas DataFrame, removes the 'Area' and 'id' columns
    if they exist, adds a new column for filenames, renames the 'label' column to 'Type' if it exists,
    adds a 'Type' column if it doesn't exist, changes the values in the 'Type' column to lowercase,
    and replaces building names in the 'Type' column.

    Parameters:
    - mask_path (str or pathlib.Path): Path to the GeoJSON file.

    Returns:
    - mask_gdf (geopandas.GeoDataFrame): Processed GeoPandas DataFrame.
    """
    # load the GeoJSON into a GeoPandas dataframe
    mask_gdf = gpd.read_file(mask_path)

    # remove 'Area' and 'id' columns
    mask_gdf.drop(columns=["Area", "id"], errors="ignore", inplace=True)

    # add new column for filenames
    mask_gdf["filename"] = mask_path.stem

    # Check if 'label' column exists
    if "label" in mask_gdf.columns:
        # rename the 'label' column to 'Type'
        mask_gdf = mask_gdf.rename(columns={"label": "Type"})

    # add a 'Type' column if it doesn't exist (should be background tiles only)
    mask_gdf["Type"] = mask_gdf.get("Type", "")

    # replace values in 'Type' column
    mask_gdf.replace(to_replace=["Service", "House"], value="Building", inplace=True)

    # change type column values to lower case
    mask_gdf["Type"] = mask_gdf["Type"].str.lower()

    # change column header to lower case
    mask_gdf = mask_gdf.rename(columns={"Type": "type"})

    return mask_gdf


def create_integer_pairings(mask_gdf, building_class_list):
    """
    Create integer pairings to building classes based on the order of classes in the list.

    This function takes a GeoPandas DataFrame containing a 'type' column with categorical
    building classes and creates integer representations of these classes based on their
    order in the provided building_class_list.

    Parameters:
    - mask_gdf (geopandas.GeoDataFrame): GeoPandas DataFrame containing the 'type' column
      with categorical building classes.
    - building_class_list (list): A list of strings representing the building classes.
      The order of classes in the list determines the integer mappings.

    Returns:
    - mask_gdf (geopandas.GeoDataFrame): GeoPandas DataFrame with an additional column
      'building_class_int' containing integer representations of the building classes.
    """
    # create integer pairings to building classes, based on order of classes in list
    building_class_numerical_lookups = dict(
        (building_class, index + 1)
        for index, building_class in enumerate(building_class_list)
    )

    # create column of integer representations of the categorical building classes
    pd.set_option("future.no_silent_downcasting", True)
    mask_gdf["building_class_int"] = mask_gdf["type"].replace(
        building_class_numerical_lookups
    )

    return mask_gdf


def generate_shapes(mask_gdf, img_dir, mask_filename):
    """
    Generate shapes for rasterization.

    This function takes a GeoPandas DataFrame containing polygons and corresponding integer values,
    a directory containing image files, and a mask filename. It converts the geometries in the GeoDataFrame
    to the coordinate reference system (CRS) of the corresponding raster image. Then, it generates shapes
    and values for rasterization using the rasterio library.

    Parameters:
    - mask_gdf (geopandas.GeoDataFrame): GeoPandas DataFrame containing polygons and integer values.
    - img_dir (str or pathlib.Path): Path to the directory containing image files.
    - mask_filename (str): Filename of the mask.

    Returns:
    - arr_to_burn (numpy.ndarray): Array to be burned onto the raster image.
    """
    # Define corresponding image filename
    image_filename = f"{mask_filename}.tif"
    image_file = img_dir.joinpath(image_filename)

    # Open raster image file
    with rio.open(image_file) as raster_tif:

        # Create a generator of geom, value pairs to use in rasterizing
        shapes = (
            (geom, value)
            for geom, value in zip(mask_gdf.geometry, mask_gdf.building_class_int)
        )

        if len(mask_gdf) == 0:
            arr_to_burn = np.zeros(raster_tif.shape)
        else:
            # Rasterize by the training labelled polygons
            arr_to_burn = features.rasterize(
                shapes=shapes,
                out_shape=raster_tif.shape,
                transform=raster_tif.transform,
                all_touched=True,
                fill=0,
            )

    return arr_to_burn


def count_unique_features(geojson_file):
    """
    Count the occurrences of unique features in a GeoJSON file.

    This function takes a GeoPandas DataFrame representing a GeoJSON file and counts
    the occurrences of unique features (types) in the file.

    Parameters:
    - geojson_file (geopandas.GeoDataFrame): GeoPandas DataFrame representing a GeoJSON file.

    Returns:
    - feature_counts (dict): A dictionary where keys are unique feature types and values are
      their respective counts.
    """
    feature_counts = {}
    for feature in geojson_file["type"]:
        if feature in feature_counts:
            feature_counts[feature] += 1
        else:
            feature_counts[feature] = 1
    return feature_counts


def add_features_to_dict(mask_path, mask_filename, mask_gdf, features_dict):
    """
    Add feature counts and averages to a dictionary.

    This function takes information about a mask, including its path, filename, GeoDataFrame,
    and a dictionary storing features. It updates the dictionary with counts of unique features
    present in the mask, such as the number of buildings and tents. If the mask represents
    background information, default counts for building and tent features are set to 0. Otherwise,
    it calculates the counts of unique features and averages their sizes.

    Parameters:
    - mask_path (str or pathlib.Path): Path to the mask file.
    - mask_filename (str): Filename of the mask.
    - mask_gdf (geopandas.GeoDataFrame): GeoPandas DataFrame containing the mask information.
    - features_dict (dict): Dictionary storing feature counts and averages.

    Returns:
    - features_dict (dict): Updated dictionary with feature counts and averages.
    """

    if mask_path.name.endswith("background.geojson"):
        unique_features = {
            "Building": 0,
            "Tent": 0,
        }
    else:
        unique_features = count_unique_features(mask_gdf)

    features_dict[mask_filename] = unique_features

    return features_dict


def rasterize_training_data(
    mask_path, mask_dir, img_dir, building_class_list, img_size, features_dict
):
    """
    Rasterize training data from GeoJSON masks.

    This function processes a GeoJSON mask file, converts it into rasterized training data,
    and saves the resulting NumPy array. It also updates a dictionary with information about
    the features present in the mask.

    Parameters:
    - mask_path (str or pathlib.Path): Path to the GeoJSON mask file.
    - mask_dir (str or pathlib.Path): Directory where the NumPy array will be saved.
    - img_dir (str or pathlib.Path): Directory containing image files.
    - building_class_list (list): A list of strings representing building classes.
    - img_size (int): Size to which the mask will be resized.
    - features_dict (dict): Dictionary storing feature counts and averages.

    Returns:
    None
    """
    # mask filename without ext.
    mask_filename = Path(mask_path).stem

    # Process GeoJSON file
    mask_gdf = process_geojson_file(mask_path)

    # Remove rows with empty geometries
    mask_gdf = mask_gdf[~mask_gdf["geometry"].is_empty]

    # Create integer pairings for building classes
    mask_gdf = create_integer_pairings(mask_gdf, building_class_list)

    # Update features dictionary with counts and averages
    features_dict = add_features_to_dict(
        mask_path, mask_filename, mask_gdf, features_dict
    )

    # Generate shapes for rasterization
    arr_to_burn = generate_shapes(mask_gdf, img_dir, mask_filename)

    # Resize to img_size
    normalised_training_arr = arr_to_burn[0:img_size, 0:img_size]

    # Save the NumPy array
    np.save(mask_dir.joinpath(f"{mask_filename}.npy"), normalised_training_arr)


def empty_geometries(geojson_path):
    with open(geojson_path, "r") as f:
        data = json.load(f)
        for feature in data["features"]:
            if "geometry" in feature and not feature["geometry"]:
                return True
    return False


def data_summary(training_data):
    """
    Reads in all .geojson files in directory 'mask_dir', checks for 'type' column,
    replaces the values 'House' and 'Service' with 'Building', and returns the
    count of each unique entry in the 'type' column for all files. Calculates the
    size of structures and returns min, max, mean for each type.

    Parameters
    ----------
    mask_dir : str or Path
        The directory containing the .geojson files to process.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the count of unique values in the 'type' column
        and the size stats for each structure type.
    """
    # create a boolean mask for rows with non-null geometries
    non_null_mask = training_data.geometry.notnull()

    # filter out rows with null geometries
    training_data_filtered = training_data[non_null_mask]

    # extract removed rows for further inspection
    removed_rows = training_data[~non_null_mask]

    # calculate structure size
    training_data_filtered["structure_area"] = training_data_filtered.geometry.apply(
        lambda geom: geom.area
    )

    # calculate statistics for each type
    structure_stats = training_data_filtered.groupby(["type"])["structure_area"].agg(
        ["min", "max", "mean"]
    )

    # calculate value counts
    value_counts = training_data_filtered.groupby(["type"])["filename"].count()

    # Return the processed data, removed rows, and summary statistics
    return training_data_filtered, value_counts, structure_stats, removed_rows
