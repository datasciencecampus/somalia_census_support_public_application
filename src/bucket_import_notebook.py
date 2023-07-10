# ## Notebook for importing data from WIP bucket
#
# > More functions including how to delete files or download the entire WIP folder (including images and masks) are in `bucket_access_functions.py`

from bucket_access_functions import (
    download_run_from_bucket,
    download_folder_from_bucket,
)
from functions_library import get_folder_paths

folder_dict = get_folder_paths()
folders = ["img", "mask", "models", "outputs"]
# work-in-progress bucket
bucket_name = folder_dict["wip_bucket"]
destination_folder = folder_dict["destination_folder"]

# ### Importing individual models and associated files.
#
# To import a specific model run - identified by `run_id`

run_id = "phase_1_gpu_1_28_06_23"

for folder_name in folders:
    destination_folder = folder_dict[folder_name + "_dir"]
    download_run_from_bucket(bucket_name, folder_name, destination_folder, run_id)

# ## Importing whole folders
#
# > Imports everything in `img`, `mask`, `models`, `outputs` folders

for folder_name in folders:
    destination_folder = folder_dict[folder_name + "_dir"]
    download_folder_from_bucket(bucket_name, folder_name, destination_folder)
