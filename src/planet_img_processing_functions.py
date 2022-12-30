from pathlib import Path
from osgeo import gdal
from datetime import datetime
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.plot import show

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

import cv2

def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    return image_src


def equalize_this(image_file, with_plot=False, gray_scale=False):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    if not gray_scale:
        b_image, g_image, r_image = cv2.split(image_src)

        band_list = [b_image, g_image, r_image]

        b_image, g_image, r_image = [np.ma.masked_where(band == 0, band) for band in band_list]

        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)

        image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        cmap_val = None
    else:
        image_eq = cv2.equalizeHist(image_src)
        cmap_val = 'gray'

    if with_plot:
        fig = plt.figure(figsize=(10, 20))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Equalized")

        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_eq, cmap=cmap_val)
        return True
    return image_eq

im_eq = equalize_this(image_file=str(tiff_img_list[0].resolve()), with_plot=False)

masked_img = np.ma.masked_where(im_eq==0, im_eq)

ax = plt.subplot(1,1,1)

ax.imshow(masked_img.data)


image_src = cv2.imread(str(tiff_img_list[0].resolve()))

plt.imshow(image_src)
 # code here largely thanks to: https://msameeruddin.hashnode.dev/image-equalization-contrast-enhancing-in-python
 # using the equalize histogram approach. Still no real luck improving contrast.

#TODO: Explore https://github.com/planetlabs/color_balance for rendering colours


# %%

# Refactor into funcs for: (not in order)
# 1) reading planet image
# 2) getting dates from title
# 3) change band order of images
# 4) unzip images
# 5) image standardisation

#%%

def return_array_from_tiff(img_path):
    """Get array from tiff raster.

    Parameters
    ----------
    img_path : Path
        Full path to img file to open.
    """
    with rio.open(img_path) as img:
        img_array = img.read()
    return(img_array)


def change_band_order(
    img_array,
    correct_band_order = [3, 2, 1, 4]
):
    """Changes the order of the raster bands. Default is to assume
    raster order BGR and correct to RGB.

    Parameters
    ----------
    img_array : numpy.ndarray
        The raster in numpy array format.
    correct_band_order : list, optional
        The correct order of bands to get RGB. Planet imagery by default is
        blue, green, red, IR. So we need to parse [3, 2, 1, 4]. This is the
        value.
    """
    img_array = [img_array[band-1] for band in correct_band_order]
    return(np.array(img_array))
# %%
img_array = return_array_from_tiff(tiff_img_list[0])

img_arr_reordered = change_band_order(img_array)

# %%

from sklearn.preprocessing import normalize

normalised_img = np.array([
    normalize(band_array, norm='max', axis=0) for band_array in img_arr_reordered
    ])

# %%
show(normalised_img)


