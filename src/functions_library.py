# ## The script contains functions to be imported and used elsewhere.

from pathlib import Path
from typing import List
import yaml
import numpy as np
import rasterio as rio
from rasterio.windows import Window

# set data directory
data_dir = Path.cwd().parent.joinpath("data")


def setup_sub_dir(data_dir: Path, sub_dir_name: str) -> Path:
    """
    Check if subdirectory of given name exists and create if not, return path.
    Parameters
    ----------
    data_dir : pathlib.Path
        Path to current data directory.
    sub_dir_name : str
        Name of desired subdirectory within data directory.
    Returns
    -------
    sub_dir : pathlib.Path
        Path to the newly created, or pre-existing, subdirectory of given name.
    """
    sub_dir = data_dir.joinpath(sub_dir_name)
    if not sub_dir.is_dir():
        sub_dir.mkdir(parents=True, exist_ok=True)
    return sub_dir


def generate_file_list(
    data_dir: Path, file_extension: str, keyword_list: list
) -> List[Path]:
    """
    Generate a list of detected files.
    Returns list of files containing given keywords, of given file extension
    in the given directory.
    Parameters
    ----------
    data_dir : pathlib.Path
        Directory to search for files in.
    file_extension : str
        The file extension of the desired files, without the dot ".".
        (e.g. "tif" or "png" or "txt").
    keyword_list : list(str)
        List of keyword(s) that should be present in selected file names.
    Returns
    -------
    file_list : list(pathlib.Path)
        List of files containing given keywords, of given file extension
        in the given directory.
    Raises
    ------
    FileNotFoundError
        Error returned if empty list generated while executing procedure.
        If this happens, check searching in the correct place and correct
        search terms are in file_extension and keyword_list.
    """
    file_list = [
        file
        for file in list(data_dir.glob(f"*.{file_extension}"))
        if all(keyword in file.name for keyword in keyword_list)
    ]
    if file_list:
        return file_list
    else:
        message = (
            f"No files were found of extension '.{file_extension}' with "
            f"{keyword_list} in the name in the directory {data_dir}."
        )
        raise FileNotFoundError(message)


def list_directories_at_path(dir_path):
    """Return list of subdirectories at given path directory."""
    return [item for item in dir_path.iterdir() if item.is_dir()]


def get_folder_paths():
    with open("../config.yaml", "r") as f:
        folder_paths = yaml.safe_load(f)
    return folder_paths


# get folder paths from config.yaml
folder_dict = get_folder_paths()
# list of folder names
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


def get_data_paths(training_dir, selected_folder):
    """
    Get paths for img and mask files sub_dir based on selected folder

    Args:
    selected_folder (str): selected folder name


    Returns:
        img_files (list): list of paths to image files
        mask_files (list): list of paths to mask files
    """
    img_dir = training_dir / selected_folder / "img"
    mask_dir = training_dir / selected_folder / "mask"

    return img_dir, mask_dir


def check_directory_images(directory):
    """
    Check the shape of .npy files in the specified directory and highlight those with incorrect shapes.

    Args:
        directory (str or Path): The directory containing the .npy files to check.
    """

    def check_array_shape(file):
        np_array = np.load(file)
        if np_array.shape != (256, 256, 4):
            print(f"File {file} has shape {np_array.shape}, which is incorrect.")

    # Get all .npy files in the directory
    array_files = list(Path(directory).glob("*.npy"))

    for file in array_files:
        check_array_shape(file)


def rm_tree(pth):

    """
    Removes everything in the chosen folder then that folder is also deleted.

    Parameters
    ----------
    pth: Path
        local data path

    Returns
    -------
    Removal of folder and it's contents from data directory
    """

    pth = Path(pth)
    if pth.exists():
        for child in pth.glob("*"):

            if child.is_file():
                child.unlink()

            else:
                rm_tree(child)

        pth.rmdir()


def delete_files_with_extensions(directory_path, extensions):
    """
    Delete files with specified extensions in a directory.

    Parameters:
        directory_path (str): The path to the directory containing the files.
        extensions (list of str): List of file extensions to delete.

    Returns:
        None
    """
    # Create a Path object for the directory
    directory = Path(directory_path)

    # Iterate through all files in the directory
    for file in directory.iterdir():
        # Check if the file extension matches any of the specified extensions
        if file.suffix in extensions:
            # Delete the file
            file.unlink()
            print(f"File '{file.name}' deleted successfully.")


def generate_tiles(image_path, tile_size=384, output_dir="tiles"):

    """
    Creates multiple tiles of planet image and saves them in specified folder.

    Parameters
    ----------
    image_path: Path
        local data path for images

    tile_size: Integer
        specified tile size

    output_dir: Path
        path for outputs to be saved in

    Returns
    -------
    Multiple tiles of planet
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rio.open(image_path) as src:
        rows = -(src.height // -tile_size)  # ceiling division
        cols = -(src.width // -tile_size)

        for row in range(rows):
            for col in range(cols):
                window = Window(col * tile_size, row * tile_size, tile_size, tile_size)
                # Read the tile
                tile = src.read(window=window)

                if np.sum(tile) > 0:
                    tile_meta = src.meta.copy()
                    tile_meta.update(
                        {
                            "width": tile_size,
                            "height": tile_size,
                            "transform": rio.windows.transform(window, src.transform),
                        }
                    )

                    tile_filename = (
                        output_dir
                        / f"{image_path.stem.replace('polygon', '')}_tile_{row}_{col}.tif"
                    )
                    with rio.open(tile_filename, "w", **tile_meta) as dst:
                        dst.write(tile)
