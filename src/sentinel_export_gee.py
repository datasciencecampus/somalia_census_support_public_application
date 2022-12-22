# Script courtesy of https://github.com/datasciencecampus/uganda_forestry/blob/master/src/python/sentinel_export_gee.py
# with minor modifications on file paths

import ee
import time
import geemap
import geopandas as gpd
from pathlib import Path
import webbrowser

ee.Initialize()


def get_s2c(aoi, start_date, end_date, img_cloud_filter):
    """Return GEE Sentinel 2 image collection with s2cloudless cloud prob band.

    Args:
        aoi (ee.Geometry): Google Earth Engine geometry bounding box for area of interest
        start_date (date (yyyy-MM-dd)): Min date for image collection images
        end_date (date (yyyy-MM-dd)): Max data for image collection images
        img_cloud_filter (int): Max amount of cloud in image metadata

    Returns:
        ee.ImageCollection : Filtered image collection of joined s2 and s2cloudless images.
    """
    # Import and filter S2 SR.
    s2c = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", img_cloud_filter))
    )

    # Import and filter s2cloudless.
    s2_cloud = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    innerJoin = ee.Join.inner()
    filterID = ee.Filter.equals(leftField="system:index", rightField="system:index")

    innerJoined = innerJoin.apply(s2c, s2_cloud, filterID)

    return ee.ImageCollection(
        innerJoined.map(lambda f: ee.Image.cat(f.get("primary"), f.get("secondary")))
    )


def get_s2_img(
    extent_fc,
    start_date,
    end_date,
    cloud_img_whole=15,
    cloud_pixel_prob=25,
    s2_bands=["B2", "B3", "B4", "B8", "B11"],
):
    """Create s2 image with selected bands for aoi.

    Args:
        extent_fc (ee.Geometry()): bounding box for image
        date_from (date (yyyy-MM-dd)): date min for composite image
        date_to (date (yyyy-MM-dd)): date max for composite image
        cloud_img_whole (int, optional): s2 image metadata whole cloudy pixel max percent. Defaults to 15.
        cloud_pixel_prob (int, optional): s2cloudless pixel probability. Defaults to 25.

    Returns:
        ee.Image
    """
    s2c = get_s2c(extent_fc, start_date, end_date, cloud_img_whole)

    def mask_cloud(img):
        mask = img.select("probability").focal_min(2).focal_max(10).lt(cloud_pixel_prob)
        return img.updateMask(mask)

    s2c = s2c.map(mask_cloud)
    s2_img = s2c.median().select(s2_bands)
    return s2_img


def get_s1_img(extent_fc, start_date, end_date, bands=["VH", "VV"]):
    """Return a Sentinel 1 SAR temporal composite image.

    Args:
        extent_fc (ee.Geometry): bounding box for image
        date_from (date (yyyy-MM-dd)): date min for composite image
        date_to (date (yyyy-MM-dd)): date max for composite image
        bands (list, optional): Selected Sentinel 1 bands of VV and VH. Defaults to ["VH"].

    Returns:
        ee.Image: Composite ee.Image for Sentinel 1 image
    """
    s1c = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(extent_fc)
        .filterDate(start_date, end_date)
    )
    s1_img = s1c.median()
    s1_img = s1_add_ratio(s1_img)
    return s1_img


def s1_add_ratio(input_image):
    """Sentinel 1 ratio band calculated by subtracting VV and VH polarisations.

    Args:
        input_image (ee.Image): Sentinel 1 GEE image with VV and VH bands.

    Returns:
        ee.Image: Sentinel 1 ee.Image with VV/VH ratio band added.
    """
    return input_image.addBands(
        input_image.select("VV").subtract(input_image.select("VH")).rename("ratio")
    )


def export_sentinel_gdrive(
    gee_img, folder_name, img_file_name, aoi, crs, poll_completion=False
):
    """Export a Google Earth Engine Image to Google Drive.

    Args:
        gee_img (ee.Image): Earth Engine image to export.
        folder_name (str): Name of the Google Drive folder in which the exported image will be created.
        img_file_name (str): Name of the output file name should not include file type extension (e.g. omit .tif)
        aoi (ee.Geometry): Earth Engine geometry representing the extent of the image to export.
        crs (string): String in EPSG:<code> format for the coordinate reference system assigned to the exported image.
        poll_completion (bool, optional): If True, keep testing for completed export until True. Defaults to False.
    """
    task = ee.batch.Export.image.toDrive(
        image=gee_img,
        folder=folder_name,
        description=img_file_name,
        region=aoi,
        scale=10,
        crs=crs,
        maxPixels=10000000000,
    )
    task.start()
    if poll_completion:
        while task.active():
            print("Polling for task (id: {}).".format(task.id))
            time.sleep(30)
        print("Done with training export.")


def get_shape_geometry_crs(input_shape_fp):
    """Read a GeoPandas data type from file and convert its extent to Earth Engine Geometry.

    Args:
        input_shape_fp (str): Path to shapefile or similar spatial dataset to be read by GeoPandas.

    Returns:
        str: EPSG code of the original shapefile coordinate reference system.
        ee.Geometry: Earth Engine geometry of the shape extent in decimal degree coordinates.
    """
    shape_gdf = gpd.read_file(input_shape_fp)
    original_crs = shape_gdf.crs.to_string()
    shape_gdf.to_crs("EPSG:4326", inplace=True)
    aoi_geom = ee.Geometry.Rectangle(*shape_gdf.total_bounds)
    return original_crs, aoi_geom


def show_gee_map(aoi_geom, image_to_show, data_dir, sentinel_satellite="sentinel2"):
    """Show a Google Earth Engine image using geemap.

    Args:
        aoi_geom (ee.Geometry): Earth Engine geometry for the area of interest where map will zoom to by default.
        image_to_show (ee.Image): Earth Engine image to display on gee map browser.
        data_dir (str): Path to the data directory which is where html for the map will be written.
        sentinel_satellite (str, optional): Options of either sentinel2 or sentinel1. Defaults to "sentinel2".

    Returns:
        None: If decide to export map otherwise quit.
    """
    # Make the template map
    preview_map = geemap.Map()
    preview_map.centerObject(aoi_geom, zoom=17)

    # Add the image layers with appropriate visualisation parameters
    if sentinel_satellite == "sentinel2":
        vis_params = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 2000}
        preview_map.addLayer(image_to_show, vis_params, "sentinel2")
    elif sentinel_satellite == "sentinel1":
        vis_params = {
            "bands": ["VV", "VH", "ratio"],
            "min": [-15, -20, 0],
            "max": [0, -5, 15],
        }
        preview_map.addLayer(image_to_show, vis_params, sentinel_satellite)

    # Save the map html temporarily (only way to display geemap outside of Jupyter Notebook)
    out_html = data_dir / "preview_map.html"
    preview_map.save(out_html)
    webbrowser.open(f"file://{out_html}")
    export_response = input("continue and export image (y/n)?")
    if export_response.lower() in ("y", "yes"):
        return None
    else:
        print("Selected not to export")
        quit()


def main(
    input_shape="sentinel2_aoi.shp",
    sentinel_satellite="sentinel2",
    start_date="2020-04-01",
    end_date="2020-05-31",
    plot_before_export=True,
):
    """Main function for sentinel Earth Engine exports.

    Args:
        input_shape (str, optional): Shapefile whose full extent will define the image export area. Must be in data dir. Defaults to "sentinel2_aoi.shp".
        sentinel_satellite (str, optional): One of sentinel1 or sentinel2. Defaults to "sentinel2".
        start_date (str, optional): Start date for the composite image exported in yyyy-MM-dd format. Defaults to "2020-04-01".
        end_date (str, optional): End date for the composite image exported in yyyy-MM-dd format. Defaults to "2020-05-31".
        plot_before_export (bool, optional): If True show a geemap of the image before exporting. Defaults to True.

    Raises:
        ValueError: If specify str other than sentinel1 sentinel2 for sentinel_satellite argument.
    """
    # Set directories
    script_path = Path(__file__).resolve()
    data_dir = script_path.parent.joinpath("data")
    input_shape_fp = script_path.parent.joinpath(input_shape)

    # Get the area of interest as GEE geometry
    output_crs, aoi_geom = get_shape_geometry_crs(input_shape_fp)

    # Get the sentinel image based on requested type
    if sentinel_satellite == "sentinel2":
        output_img = get_s2_img(aoi_geom, start_date, end_date)
    elif sentinel_satellite == "sentinel1":
        output_img = get_s1_img(aoi_geom, start_date, end_date)
    else:
        raise ValueError(
            "Must specifiy sentinel2 or sentinel1 for sentinel_satellite parameter"
        )

    # Show map of image before export
    if plot_before_export:
        show_gee_map(aoi_geom, output_img, data_dir, sentinel_satellite)
    output_file_name = (
        f"{Path(input_shape).stem}_{sentinel_satellite}_{start_date}_{end_date}"
    )
    # TO DO add exporting
    export_sentinel_gdrive(
        gee_img=output_img,
        folder_name="forestry_uganda",
        img_file_name=output_file_name,
        aoi=aoi_geom,
        crs=output_crs,
        poll_completion=False,
    )


def mk_arg_pars():
    """Create a comand line arg parse.
    Returns:
        dict: Argparse argument dictionary.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a Sentinel 2 or Sentinel 1 (SAR) image using Google Earth Engine API."
    )
    parser.add_argument(
        "-i",
        "--input-shape",
        default="input/analysis_zones.shp",
        help="Shapefile whose full extent will define the image export area. Must be in repo folder input or data. Defaults to input/analysis_zones.shp",
    )

    parser.add_argument(
        "-s",
        "--sentinel-satellite",
        default="sentinel2",
        help="Sentinel satellite which must be sentinel2 or sentinel1. Default sentinel 2.",
    )
    parser.add_argument(
        "-d1",
        "--start-date",
        default="2021-01-01",
        help="Start date for the composite image exported in yyyy-MM-dd format. Defaults to 2021-01-01",
    )

    parser.add_argument(
        "-d2",
        "--end-date",
        default="2021-03-31",
        help="End date for the composite image exported in yyyy-MM-dd format. Defaults to 2021-03-31",
    )

    parser.add_argument(
        "-p",
        "--plot-before-export",
        action="store_true",
        help="Include the -p parameter to plot the image and then be prompted to continue export or cancel after viewing.",
    )

    args_pars = parser.parse_args()
    return vars(args_pars)


if __name__ == "__main__":
    run_dict = mk_arg_pars()
    main(**run_dict)
