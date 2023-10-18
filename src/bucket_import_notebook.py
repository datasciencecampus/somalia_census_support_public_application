# ## Notebook for importing data from WIP bucket
#
# **Purpose**
#
# To import files or folders from the WiP bucket to local GCP storage
#
# **Things to note**
#
# - Do not run all cells in this notebook, instead choose the function you want to perform (import a file or folder) and run only that action

# ### Set-up

# +
from pathlib import Path

from bucket_access_functions import (
    download_run_from_bucket,
    download_folder_from_bucket,
    read_files_in_bucket,
)
from functions_library import get_folder_paths

# +
# local folders
folder_dict = get_folder_paths()
folders = [
    "training_img_dir",
    "training_mask_dir",
    "validation_img_dir",
    "validation_mask_dir",
    "ramp_mask",
    "ramp_img",
    "models_dir",
    "outputs_dir",
]

(
    training_img,
    training_mask,
    validation_img,
    validation_mask,
    ramp_mask,
    ramp_img,
    models,
    outputs,
) = [Path(folder_dict[folder]) for folder in folders]

model_folders = ["models_dir", "outputs_dir"]

# work-in-progress bucket
bucket_name = folder_dict["wip_bucket"]
# -

# ### Read in files in bucket

read_files_in_bucket(bucket_name)

# ### Import individual files

# #### Importing individual models and associated files.
#
# >To import a specific model run - identified by `run_id`

run_id = "phase_1_gpu_test_no_ga"

for folder in model_folders:
    destination_folder = folder_dict[folder]
    download_run_from_bucket(bucket_name, folder, destination_folder, run_id)

# #### To download just one file

# +
folder_name = "to_set"
destination_folder = "to_set"
file_name = "to_set"

download_run_from_bucket(bucket_name, folder_name, destination_folder, file_name)
# -

# ### Importing whole folders
#

# #### Importing all models and associated folders
#
# > Imports everything in `img`, `mask`, `models`, `outputs` folders

for folder in model_folders:
    destination_folder = folder_dict[folder]
    download_folder_from_bucket(bucket_name, folder_name, destination_folder)

# #### Importing a single folder

# +
folder_name = "ramp_bentiu_south_sudan/img"
destination_folder = ramp_img

download_folder_from_bucket(bucket_name, folder_name, destination_folder)
