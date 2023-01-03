# ## The script contains functions to be imported and used elsewhere.

from pathlib import Path
from typing import List

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


def create_geojsons_of_extents(
    area_to_process,
    data_directory_path,
    full_shapefiles_path = None
):
    # Convert shapefile geometries into geojson files.
    if full_shapefiles_path:
        shapefile = full_shapefiles_path
    elif not full_shapefiles_path:
        shapefile = data_directory_path.joinpath(
        "IDP Priority Area Extent Shapefiles",
        "IDP Priority Area Extent Shapefiles",
        "IDP Survey Shapefiles",
        f"{area_to_process}_Extent.shp"
        )
    shapefile_gdf = geopandas.read_file(shapefile)
    shapefile_gdf.to_file(priority_area_geojsons_dir.joinpath(f"{area_to_process}_extent.geojson"), driver='GeoJSON')


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
    return([item for item in dir_path.iterdir() if item.is_dir()])
