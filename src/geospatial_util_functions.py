""" Library of general geospatial related functions """

import geopandas as gpd
import rasterio as rio
from pyproj import Transformer


def convert_shapefile_to_geojson(
    shapefile_file_path, output_dir, output_file_name=None
):
    if not output_file_name:
        output_file_name = shapefile_file_path.stem
    # Convert shapefile geometries into geojson files and save outputs.
    shapefile_gdf = gpd.read_file(shapefile_file_path)
    shapefile_gdf.to_file(
        output_dir.joinpath(f"{output_file_name}.geojson"), driver="GeoJSON"
    )


# TODO: Add check for existing files and ignore if present.


def return_geo_meta_from_tiff(img_path):
    """Get bounding box and CRS from tiff raster.

    Parameters
    ----------
    img_path : Path
        Full path to img file to open.
    """
    with rio.open(img_path) as raster:
        bounding_box = raster.bounds
        crs = raster.crs
    return (bounding_box, crs)


def get_reprojected_bounds(raster_file, desired_crs=4326):
    """Returns bounds from raster and reprojects to desired CRS"""
    bounding_box, raster_crs = return_geo_meta_from_tiff(raster_file)
    xmin, ymin, xmax, ymax = bounding_box
    transformer = Transformer.from_crs(raster_crs, desired_crs)
    xmin, ymin = transformer.transform(xmin, ymin)
    xmax, ymax = transformer.transform(xmax, ymax)
    bounding_box = [[xmin, ymin], [xmax, ymax]]
    return bounding_box


def check_crs_and_reset(shapefile, desired_crs="EPSG:4326"):
    """Check the CRS of a geospatial file and if not desired CRS, set to CRS and overwrite.

    Parameters
    ----------
    shapefile : path
        File path to geospatial to work with (such as shapefile).
    desired_crs : str, optional
        The desired CRS in format "EPSG:<epsg_number>. The default is "EPSG:4326"
        corresponding to the WSG84 lon & lat system.
    """
    gdf_for_aoi = gpd.read_file(shapefile)
    if gdf_for_aoi.crs.to_string() != desired_crs:
        print(f"Converting CRS for shapefile {shapefile.stem}")
        gdf_for_aoi = gdf_for_aoi.to_crs(crs=desired_crs)
        gdf_for_aoi.to_file(shapefile)
