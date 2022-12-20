from pathlib import Path
from osgeo import gdal
from datetime import datetime
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np

from functions_library import setup_sub_dir, generate_file_list

def extract_dates_from_image_filenames(file_name):
    """
    Extract date information from Planet imagery file names.

    e.g. file named 'Doolow_W_50MB_20221101' would return
    datetime.datetime(2022, 11, 1, 0, 0)

    Parameters
    ----------
    file_name : str
        The file name for given Planet image.

    Returns
    -------
    datetime.datetime
        Date data type from string.
    """
    numerical_string = file_name.split("_")[-1]
    date = datetime.strptime(numerical_string, "%Y%m%d")
    return date


def unzip_image_files(file_to_unzip):
    # opening the zip file in READ mode
    with ZipFile(file_to_unzip, 'r') as zip:
        # extracting all the files
        zip.extractall(setup_sub_dir(file_to_unzip.parent, file_to_unzip.stem))
        print(f"Files extracted {file_to_unzip.parent.joinpath(file_to_unzip.stem)}")


data_dir = Path.cwd().parent.joinpath("data")

planet_imgs_path = setup_sub_dir(data_dir, "planet_images")

path_to_imgs = planet_imgs_path.joinpath("Doolow")

zipped_observation_list = generate_file_list(
        path_to_imgs, "zip", []
    )

zipped_files = [file_name.stem for file_name in zipped_observation_list]

observation_dates = [
    extract_dates_from_image_filenames(file_name) for file_name in zipped_files
    ]

for observation in zipped_observation_list:
    dir_path = observation.parent.joinpath(observation.stem)

    # if directory does not exist, unzip it first
    if not dir_path.is_dir():
        unzip_image_files(observation)

    tiff_img_list = generate_file_list(
            dir_path.joinpath("files"), "tif", ["pansharpened_clip"]
            )

    planet_img = gdal.Open(str(tiff_img_list[0].resolve()))


#%%
from sklearn.preprocessing import normalize

band_number_list = [3, 2, 1] # red, green, blue in order Planet uses

band_list = [planet_img.GetRasterBand(band_number) for band_number in band_number_list]

band_array_list = [band.ReadAsArray() for band in band_list]

normalised_bands = [
    normalize(band_array, norm='max', axis=0) for band_array in band_array_list
    ]

col_img = np.dstack((normalised_bands))
f = plt.figure()
plt.imshow(col_img)
plt.show()

#TODO: Explore https://github.com/planetlabs/color_balance for rendering colours


# %%





