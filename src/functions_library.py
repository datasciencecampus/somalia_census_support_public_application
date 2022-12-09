### The script contains functions to be imported and used elsewhere.

from pathlib import Path

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


def create_geojsons_of_extents(area_to_process, data_directory_path):
    # Convert shapefile geometries into geojson files.
    shapefile = data_directory_path.joinpath(
      "IDP Priority Area Extent Shapefiles",
      "IDP Priority Area Extent Shapefiles",
      "IDP Survey Shapefiles",
      f"{area_to_process}_Extent.shp"
     )
    shapefile_gdf = geopandas.read_file(shapefile)
    shapefile_gdf.to_file(priority_area_geojsons_dir.joinpath(f"{area_to_process}_extent.geojson"), driver='GeoJSON')