""" Script of functions related to model training preprocessing. """

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features


def rasterize_training_data(
    training_data: gpd.GeoDataFrame,
    reference_satellite_raster: Path,
    building_class_list: list,
    binary_classify=False,
):
    """Generate segmented raster of training data from polygons.

    For a given training set, recreate the extent of the corresponding satellite
    raster but with pixels valued by training labels. For example, pixels labelled
    as buildings set to one, tents set to two and background set to zero. The
    actual building classes and numerical values depends on building_class_list
    parameter and its order.

    Outputs and saves the generated raster as a tif file.

    Parameters
    ----------
    training_data : gpd.GeoDataFrame
        The training data geometry data, loaded in geopandas.
    reference_satellite_raster : Path
        File path to the reference raster corresponding to the given training data.
    building_class_list : list
        A list of the classes of building present in the training data.
    output_tif_file_path : Path
        File path to desired output segmented training tif file.
    binary_classify : bool
        Determines whether to rasterize on multiple classes or just a binary
        building / non-building set-up. Default is False and distinctive building
        classes are assigned different pixel values.

    """
    if binary_classify:
        # assign all polygons with value 1 regardless of building class
        building_class_numerical_lookups = dict(
            (building_class, 1)
            for index, building_class in enumerate(building_class_list)
        )
    else:
        # create integer pairings to building classes, based on order of classes in list
        building_class_numerical_lookups = dict(
            (building_class, index + 1)
            for index, building_class in enumerate(building_class_list)
        )
    # open corresponding satellite raster
    raster_tif = rio.open(reference_satellite_raster)

    # create column of integer representations of the categorical building classes
    training_data["building_class_int"] = training_data["Type"].replace(
        building_class_numerical_lookups
    )
    # ensure the CRS for the training data and satellite raster match
    training_data = training_data.to_crs(raster_tif.crs)

    # create a generator of geom, value pairs to use in rasterizing
    shapes = (
        (geom, value)
        for geom, value in zip(training_data.geometry, training_data.building_class_int)
    )
    # check if any polygons present - if not, i.e. background tile, then generate zero array
    if len(training_data) == 0:
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
    # return the array representation
    return arr_to_burn


def check_mask_files(mask_dir, ref_shape=(384, 384)):
    """
    Check all .npy files in the given directory against a reference shape.

    Args:
    img_dir(str or pathlib.Path): Path to the directory containing the image files.
    ref_shape (tuple of int, optional): The reference shape that each image should have.
        Defaults to (384, 384).

    Raises:
        Warning: If an image file has a different shape than the reference shape.
    """
    mask_file_list = mask_dir.glob("*npy")
    for file in mask_file_list:
        mask_array = np.load(file)
        if mask_array.shape != ref_shape:
            warnings.warn(f"{file} has a different shape than the reference shape")


def training_data_summary(mask_dir):
    """
    Reads in all .geojson files in directory 'mask_dir', checks for 'Type' column,
    replaces the values 'House' and 'Service' with 'Building', and returns the
    count of each unique entry in the 'Type' column for all files.

    Parameters
    ----------
    mask_dir : str or Path
        The directory containing the .geojson files to process.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the count of unqiue values in the 'Type' column
    """
    # empty dataframe
    training_data = None

    for file in mask_dir.glob("*.geojson"):

        # read GeoJSON into GeoDataFrame
        df = gpd.read_file(file)

        # check for 'type' column
        if "Type" not in df.columns:
            df["Type"] = ""

        # replace values in 'Type' column
        df["Type"].replace({"House": "Building", "Service": "Building"}, inplace=True)

        if training_data is None:
            training_data = df
        else:
            training_data = training_data.append(df)

    # return value counts
    value_counts = training_data["Type"].value_counts()

    return training_data, value_counts
