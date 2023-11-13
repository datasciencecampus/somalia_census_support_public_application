#!/usr/bin/env python
# coding: utf-8
# %%
""" Script of functions for checking mask and img files before they are added GCP ingress folder. """


# %%
from pathlib import Path
from functions_library import get_folder_paths
import re
import warnings
import geopandas as gpd
from colorama import Fore


# %%
# Note directories of interest
folder_dict = get_folder_paths()

training_img_dir = Path(folder_dict["training_img_dir"])
training_mask_dir = Path(folder_dict["training_mask_dir"])
validation_img_dir = Path(folder_dict["validation_img_dir"])
validation_mask_dir = Path(folder_dict["validation_mask_dir"])


# %%
def change_to_lower_case(files):

    """
    Changes file names for img and mask files to lower case in training or validation data before ingress to GCP.

    Parameters
    ----------
    files: list
        Absolute paths for img or mask files on local machine

    Returns
    -------
    files_lower : list
        Lower case img and mask file names stored in list.
    """

    # Makes img and mask file names lower case and stored in list
    files_lower = [Path(file.parent, file.name.lower()) for file in files]

    # Loop through both lists to rename original img or mask file name
    for file_original, file_lower in zip(files, files_lower):
        file_original.rename(file_lower)

    return files_lower


# %%
def vice_versa_check_mask_file_for_img_file(img_files, mask_files, for_mask_or_img):

    """
    Checks mask file names to see if they have a corresponding img file in training or validation data before ingress to GCP.
    Will also check if img file names to see if they have a corresponding mask file

    Parameters
    ----------
    img_files: list
        List of img files in lower case
    mask_files: list
        List of mask files in lower case
    for_mask_or_img: character
        If "mask" then will check whether mask files have an img file with the same name. If "img" will do
        check whether img files have a mask file with the same name.

    Returns
    -------
    Warning if there is not a corresponding img file for mask file name or vice versa
    """

    # Note whether working on mask or image files
    files_of_interest = None
    paired_files = None
    if for_mask_or_img == "mask":
        files_of_interest = mask_files
        paired_files = img_files
    elif for_mask_or_img == "img":
        files_of_interest = img_files
        paired_files = mask_files
    else:
        raise Exception(
            f'Option for for_mask_or_img paramater ({for_mask_or_img})) not recognised. Use either "mask" or "img"'
        )

    # Get names of paired files
    paired_file_names = [file.name for file in paired_files]

    # Examine each file of interest
    for file in files_of_interest:

        # Initialise string to store
        file_pair = None

        # Check if banding present (relevant for image files)
        if "bgr" in file.name or "rgb" in file.name:
            file_pair = file.stem[:-4] + ".geojson"

        elif for_mask_or_img == "img":
            file_pair = file.stem + ".geojson"
            warnings.warn(f"banding pattern isn't present in {file.name}")

        else:  # files with no banding should be mask files (GeoJSON)
            file_pair = file.stem + "_bgr.tif"

        # Check if equivalent file exists
        if file_pair not in paired_file_names:
            warnings.warn(
                f"Equivalent ({file_pair}) for current file ({file}) doesn't exist"
            )


# %%
def check_naming_convention_upheld(
    img_files_lower,
    mask_files_lower,
    data_type,
    naming_convention_pattern_for_training=r"training_data_.+_[0-9]+_*",
    naming_convention_pattern_for_validation=r"validation_data_.+_[0-9]+_*",
    naming_convention=[
        "training_data_<area>_<tile no>_<initials>_<bgr>.tif",
        "training_data_<area>_<tile no>_<initials>.geojson",
        "validation_data_<area>_<tile no>_<initials>_<bgr>.tif",
        "validation_data_<area>_<tile no>_<initials>.geojson",
    ],
):
    """
    Checks if the correct naming convention is being used for image and mask files in training or validation data.

    Parameters
    ----------
    img_files_lower: list
        List of image filenames in lowercase.
    mask_files_lower: list
        List of mask filenames in lowercase.
    data_type: str
        Indicates whether the checked tiles are for "training" or "validation" data.
    naming_convention_pattern_for_training: str, optional
        Regular expression pattern representing the expected structure for training data filenames.
    naming_convention_pattern_for_validation: str, optional
        Regular expression pattern representing the expected structure for validation data filenames.
    naming_convention: list of str
        List of strings representing the expected naming conventions.

    Returns
    -------
    None
        Generates a warning if naming convention for image or mask file is incorrect.
    """
    naming_pattern = (
        naming_convention_pattern_for_training
        if data_type == "training"
        else naming_convention_pattern_for_validation
    )

    for file in img_files_lower + mask_files_lower:
        if not re.match(naming_pattern, file.name):
            convention_type = "imgs" if data_type == "training" else "masks"
            correct_convention = (
                naming_convention[0]
                if data_type == "training"
                else naming_convention[2]
            )
            warning_message = f"The naming convention for ({file}) is not correct. Please change to {correct_convention} for {convention_type}."
            warnings.warn(warning_message)


# %%
def cleaning_of_mask_files(mask_files_lower, data_type):

    """
    Cleans geopandas dataframes of all mask files and then overwrites them in the mask folder. Checks for
    additional columns, missing columns and null values.

    Parameters
    ----------
    mask_files_lower: list
        List of mask files in lower case
    data_type: str
        Indicates whether the checked tiles are for "training" or "validation" data.

    Returns
    -------
    Warning message if "Type" column not found along with creation of new column in geopandas dataframe,
    print message and lastly a GeoJSON file that has been cleaned.
    """

    mask_dir = training_mask_dir if data_type == "training" else validation_mask_dir

    # Examine each mask file
    for mask_file in mask_files_lower:

        # Load mask
        mask_gdf = gpd.read_file(str(mask_file))
        column_names = mask_gdf.columns

        # drop fid and id columns
        if "id" in column_names:
            mask_gdf = mask_gdf.drop(columns=["id"])
        elif "fid" in column_names:
            mask_gdf = mask_gdf.drop(columns=["fid"])

        # check if type column not present and send warning
        if "Type" not in column_names and len(mask_gdf.geometry) == 0:
            warnings.warn(
                Fore.GREEN
                + f"""{(mask_file)} contains no Type or geometry.
                Ensure this mask is for a background tile. File has not been modified!"""
            )
            continue

        # check if type column not present and geometry column is - send error
        elif "Type" not in column_names and "geometry" in column_names:
            warnings.warn(
                Fore.GREEN
                + f"""{(mask_file)} contains no type but has geometry.
                Add type information to each polygon using your GIS software. File has not been saved!"""
            )
            continue

        # check building types in type column
        if len(mask_gdf.Type.unique()) == 1:
            warnings.warn(
                Fore.GREEN
                + f"""{(mask_file)} contains only 1 type of building!
                Ensure this is correct before uploading to the ingress folder."""
            )

        # check any null values in type column - send error
        if mask_gdf["Type"].isnull().values.any():
            warnings.warn(
                Fore.GREEN
                + f"Type column for ({mask_file}) has null values. File has not been saved!"
            )
            continue

        # check any null values in geometry column - send error
        if mask_gdf["geometry"].isnull().values.any():
            warnings.warn(
                Fore.GREEN
                + f"Geometry column for ({mask_file}) has null values. File has not been saved! Check the file in your GIS software."
            )
            continue

        # write back to geojson for training
        mask_gdf.to_file(mask_dir.joinpath(f"{(mask_file.name)}"), driver="GeoJSON")


# %%
def check_same_number_of_files_present(img_files, mask_files):

    """
    Checks if same number of imgs and mask files present - if not then warning

    Parameters
    ----------
    img_files: pathlib
        path for img directory for training or validation data
    mask_files: pathlib
        path for mask directory for training or validation data

    Returns
    -------
    Warning message if number of validation/training img files doesn't
    match number of validation/training mask files
    """

    # Check that same number of imgs and mask files present - if not then warning
    if len(img_files) != len(mask_files):
        warnings.warn(
            f"Number of validation image files {len(img_files)} doesn't match number of validation mask files {len(mask_files)}"
        )

    return


# %%

folder_name = [
    "training_img_dir",
    "training_mask_dir",
    "validation_img_dir",
    "validation_mask_dir",
]

# set folder paths
training_img_dir, training_mask_dir, validation_img_dir, validation_mask_dir = [
    Path(folder_dict[folder]) for folder in folder_name
]


def get_file_paths(data_type):
    """
    Get paths for image and mask files based on whether data is training or validation.

    Args:
    data_type (str): either 'training' or 'validation'

    Returns:
        img_files (list): list of paths to image files
        mask_files (list): list of paths to mask files
    """
    if data_type == "training":
        img_dir = training_img_dir
        mask_dir = training_mask_dir
    elif data_type == "validation":
        img_dir = validation_img_dir
        mask_dir = validation_mask_dir
    else:
        raise ValueError("Invalid data_type value")

    img_files = list(img_dir.glob("*.tif"))
    mask_files = list(mask_dir.glob("*.geojson"))

    return img_files, mask_files
