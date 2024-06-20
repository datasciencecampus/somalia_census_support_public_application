import pandas as pd
import numpy as np
from pathlib import Path
from rasterio import features
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio as rio

from image_processing_functions import process_image


def extract_transform_from_directory(directory):
    """
    Extract filenames and transformation matrices from GeoTIFF files in a directory.

    Parameters:
    - directory (str): Path to the directory containing GeoTIFF files.

    Returns:
    dict: Dictionary containing filename and corresponding transformation matrix.
    """
    transform_dict = {}
    directory_path = Path(directory)
    for filepath in directory_path.glob("*.tif"):
        with rio.open(filepath) as src:
            transform_dict[filepath.name] = src.transform
    return transform_dict


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


def extract_shapes_from_masks(mask_dict, transform):
    """
    Extract shapes from a dictionary of masks.

    Parameters:
    - mask_dict (dict): Dictionary containing masks.

    Returns:
    dict: Dictionary containing shapes extracted from masks.
    """
    return {
        title: features.shapes(mask, mask=mask, transform=transform, connectivity=4)
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


def process_tile(model, tile, unique_classes, filename, index_number, transforms, crs):
    """
    Process a single tile by generating masks, extracting shapes, and creating GDF.

    Parameters:
    - model: The trained model for prediction.
    - tile (np.ndarray): Input image tile.
    - unique_classes (list): List of unique class labels.
    - filename (str): Filename associated with the tile.
    - index_number (int): Index number associated with the tile.
    - transforms (dict): Dictionary containing transformation matrices.
    - crs: Coordinate reference system for GeoDataFrame.

    Returns:
    geopandas.GeoDataFrame or None: Combined GeoDataFrame containing buildings and tents information,
    or None if no shapes are detected.
    """
    test_img_input = np.expand_dims(tile, 0)
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    # Get transformation matrix
    transform = transforms.get(filename + ".tif")

    if transform is None:
        # Provide a default transform if not found
        transform = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # Default identity transform

    # Create mask dictionary
    mask_dict = create_mask_dict(predicted_img, unique_classes)

    # Extract shapes from masks
    shapes_dict = extract_shapes_from_masks(mask_dict, transform)

    # Extract shapes for buildings and tents
    buildings_shapes = shapes_dict.get("mask_1", None)
    tents_shapes = shapes_dict.get("mask_2", None)

    if buildings_shapes is None and tents_shapes is None:
        print(f"Ignoring file: {filename} - No shapes or polygons detected.")
        return None

    # Convert shapes to GDF if shapes exist
    buildings_geometries = (
        shapes_to_geopandas(buildings_shapes, "buildings", filename, index_number)
        if buildings_shapes
        else []
    )

    tents_geometries = (
        shapes_to_geopandas(tents_shapes, "tents", filename, index_number)
        if tents_shapes
        else []
    )

    # Create GDF for buildings and tents if shapes exist
    if buildings_geometries or tents_geometries:
        buildings_gdf = gpd.GeoDataFrame(buildings_geometries)
        tents_gdf = gpd.GeoDataFrame(tents_geometries)

        # Concatenate GDF
        combined_gdf = gpd.GeoDataFrame(
            pd.concat([buildings_gdf, tents_gdf], ignore_index=True)
        )

        combined_gdf.set_geometry("geometry", inplace=True)
        combined_gdf.crs = crs

        return combined_gdf
    else:
        return None


def process_image_files(img_files, img_size, sub_area_dir):
    """
    Process a list of image files, calling the process_image function on each one.

    Args:
        img_files (list): List of image file paths.
        img_size (tuple): Tuple specifying the size of the images.
        sub_area_dir (str): Directory path where the images are located.

    Returns:
        list: List of file paths that encountered errors during processing.
    """
    error_files = []
    for img_file in img_files:
        try:
            process_image(img_file, img_size, sub_area_dir)
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")
            error_files.append(img_file)
    print("Files with errors:", error_files)
    return error_files


def check_shapes(directory):
    """
    Check the shapes of npy files in a directory and print any inconsistencies.

    Args:
        directory (str): Path to the directory containing npy files.

    Returns:
        set: Set containing the shapes of npy files if they are all the same, None otherwise.
    """
    error_files = []
    npy_files = Path(directory).glob("*.npy")
    shapes = set()

    for file in npy_files:
        array = np.load(file)
        if array.shape != (384, 384, 4):
            print(f"File {file} has shape {array.shape}, expected (384, 384, 4)")
            error_files.append(file)
        else:
            shapes.add(array.shape)

    if len(shapes) == 1:
        print("All npy files have the same shape:", shapes.pop())
    else:
        return None


def is_array_empty(arr):
    """
    Check if a NumPy array is empty (all elements are zero).

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        bool: True if the array is empty, False otherwise.
    """
    return np.all(arr == 0)


def filter_empty_arrays(padded_unseen_images, unseen_filenames):
    """
    Filter out empty arrays and corresponding filenames from the input arrays.

    Args:
        padded_unseen_images (numpy.ndarray): Array of images.
        unseen_filenames (list): List of filenames corresponding to the images.

    Returns:
        tuple: A tuple containing the filtered images array and filtered filenames list.
    """
    empty_indices = [
        i for i, arr in enumerate(padded_unseen_images) if is_array_empty(arr)
    ]
    unseen_filtered = np.delete(padded_unseen_images, empty_indices, axis=0)
    filenames_filtered = [
        filename
        for i, filename in enumerate(unseen_filenames)
        if i not in empty_indices
    ]
    print("Filtered stacked array shape:", unseen_filtered.shape)
    print("Number of filtered filenames:", len(filenames_filtered))
    return unseen_filtered, filenames_filtered
