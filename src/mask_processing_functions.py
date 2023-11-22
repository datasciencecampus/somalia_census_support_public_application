""" Script of functions related to model training preprocessing. """


from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features


def process_geojson_file(mask_path):
    # load the GeoJSON into a GeoPandas dataframe
    mask_gdf = gpd.read_file(mask_path)

    # Check if 'label' column exists
    if "label" in mask_gdf.columns:
        # rename the 'label' column to 'Type'
        mask_gdf = mask_gdf.rename(columns={"label": "Type"})

    # add a 'Type' column if it doesn't exist (should be background tiles only)
    mask_gdf["Type"] = mask_gdf.get("Type", "")

    # replace values in 'Type' column
    mask_gdf["Type"].replace({"House": "Building", "Service": "Building"}, inplace=True)

    # change to lower case
    mask_gdf["Type"] = mask_gdf["Type"].str.lower()

    return mask_gdf


def create_integer_pairings(mask_gdf, building_class_list):
    # create integer pairings to building classes, based on order of classes in list
    building_class_numerical_lookups = dict(
        (building_class, index + 1)
        for index, building_class in enumerate(building_class_list)
    )

    # create column of integer representations of the categorical building classes
    mask_gdf["building_class_int"] = mask_gdf["Type"].replace(
        building_class_numerical_lookups
    )

    return mask_gdf


def generate_shapes(mask_gdf, img_dir, mask_filename):
    # define corresponding image filename
    image_filename = f"{mask_filename}.tif"
    image_file = img_dir.joinpath(image_filename)

    raster_tif = rio.open(image_file)
    mask_gdf = mask_gdf.to_crs(raster_tif.crs)

    # create a generator of geom, value pairs to use in rasterizing
    shapes = (
        (geom, value)
        for geom, value in zip(mask_gdf.geometry, mask_gdf.building_class_int)
    )
    if len(mask_gdf) == 0:
        arr_to_burn = np.zeros(raster_tif.shape)
    else:
        # rasterize by the training labelled polygons
        arr_to_burn = features.rasterize(
            shapes=shapes,
            out_shape=raster_tif.shape,
            transform=raster_tif.transform,
            all_touched=True,
            fill=0,
        )
    return arr_to_burn


def rasterize_training_data(
    mask_path, mask_dir, img_dir, building_class_list, img_size, features_dict
):
    # mask filename without ext.
    mask_filename = Path(mask_path).stem

    mask_gdf = process_geojson_file(mask_path)
    mask_gdf = create_integer_pairings(mask_gdf, building_class_list)
    features_dict = add_features_to_dict(
        mask_path, mask_filename, mask_gdf, features_dict
    )
    arr_to_burn = generate_shapes(mask_gdf, img_dir, mask_filename)

    # re-sizing to img_size
    normalised_training_arr = arr_to_burn[0:img_size, 0:img_size]

    # save the NumPy array
    np.save(mask_dir.joinpath(f"{mask_filename}.npy"), normalised_training_arr)


def add_features_to_dict(mask_path, mask_filename, mask_gdf, features_dict):
    # Add Feature counts for this tiles into features dictionary
    if mask_path.name.endswith("background.geojson"):
        unique_features = {
            "Building": 0,
            "Tent": 0,
            "Avg_building_size": 0,
            "Avg_tent_size": 0,
        }
    else:
        unique_features = count_unique_features(mask_gdf)
        tent_avg, building_avg = calculate_average_feature_size(mask_gdf)
        unique_features["tent_average"] = tent_avg
        unique_features["building_average"] = building_avg

    features_dict[mask_filename] = unique_features

    return features_dict


def data_summary(training_data):
    """
    Reads in all .geojson files in directory 'mask_dir', checks for 'Type' column,
    replaces the values 'House' and 'Service' with 'Building', and returns the
    count of each unique entry in the 'Type' column for all files. Calculates the
    size of structures and returns min, max, mean for each type.

    Parameters
    ----------
    mask_dir : str or Path
        The directory containing the .geojson files to process.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the count of unqiue values in the 'Type' column
        and the size stats for each structure type.
    """

    # return value counts
    value_counts = training_data["Type"].value_counts()

    # calculate structure size
    training_data["structure_area"] = training_data.geometry.apply(
        lambda geom: geom.area
    )

    # calculate statistics for each type
    structure_stats = training_data.groupby("Type")["structure_area"].agg(
        ["min", "max", "mean"]
    )

    return training_data, value_counts, structure_stats


def count_unique_features(geojson_file):
    feature_counts = {}
    for feature in geojson_file["Type"]:
        if feature in feature_counts:
            feature_counts[feature] += 1
        else:
            feature_counts[feature] = 1
    return feature_counts


def calculate_average_feature_size(geojson_file_df):
    tent_size = []
    building_size = []

    for idx, row in geojson_file_df.iterrows():
        size = row.geometry.area
        if row["Type"] == "Tent":
            tent_size.append(size)
        elif row["Type"] == "Building":
            building_size.append(size)

    average_tent = sum(tent_size) / len(tent_size) if tent_size else 0
    average_building = sum(building_size) / len(building_size) if building_size else 0

    return average_tent, average_building
