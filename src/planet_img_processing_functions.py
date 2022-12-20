from pathlib import Path
from datetime import datetime

from functions_library import setup_sub_dir, generate_file_list

data_dir = Path.cwd().parent.joinpath("data")

planet_imgs_path = setup_sub_dir(data_dir, "planet_images")

path_to_imgs = planet_imgs_path.joinpath("Doolow")

observation_list = generate_file_list(
        path_to_imgs, "zip", []
    )

file_names = [file_name.stem for file_name in observation_list]

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

observation_dates = [
    extract_dates_from_image_filenames(file_name) for file_name in file_names
    ]

planet_img = gdal.Open(str(path_to_imgs.resolve()))

