""" Script of functions related to model training preprocessing. """

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features


def rasterize_training_data(
    training_data: gpd.GeoDataFrame,
    reference_satellite_raster: Path,
    building_class_list: list,
    output_tif_file_path,
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
    # retrieve geospatial meta data from corresponding raster
    raster_meta = raster_tif.meta.copy()

    # create column of integer representations of the categorical building classes
    training_data["building_class_int"] = training_data["Type"].replace(
        building_class_numerical_lookups
    )
    # ensure the CRS for the training data and satellite raster match
    training_data = training_data.to_crs(raster_tif.crs)

    with rio.open(output_tif_file_path, "w+", **raster_meta) as out:
        out_arr = out.read(1)
        # create a generator of geom, value pairs to use in rasterizing
        shapes = (
            (geom, value)
            for geom, value in zip(
                training_data.geometry, training_data.building_class_int
            )
        )
        # rasterize by the training labelled polygons
        arr_to_burn = features.rasterize(
            shapes=shapes,
            out=out_arr,
            transform=out.transform,
            all_touched=True,
            fill=0,
        )
        # save raster as tif
        out.write_band(1, arr_to_burn)
        # return the array representation
        return arr_to_burn


def reorder_array(img_arr, height_index, width_index, bands_index):
    # Re-order the array into height, width, bands order.
    arr = np.transpose(img_arr, axes=[height_index, width_index, bands_index])
    return arr
