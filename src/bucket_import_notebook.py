# ## Notebook for importing data from WIP bucket
#
# <div class="alert alert-block altert-danger">
#     <i class="fa fa-exclamation-triangle"></i> check the kernel in the above right is `python3` <b>not</b> `venv-somalia-gcp`
# </div>
#
# **Purpose**
#
# To import files or folders from the WiP bucket to local GCP storage
#
# **Things to note**
#
# - Do not run all cells in this notebook, instead choose the function you want to perform (import a file or folder) and run only that action
#
# #### Jump to:
# 1. ##### [Read files in bucket](#read)
# 1. ##### [Importing individual files](#importfiles)
# 1. ##### [Importing folders](#importfolders)
# 1. ##### [Delete files in bucket](#delete)
#

# ### Set-up

from bucket_access_functions import (
    download_run_from_bucket,
    download_folder_from_bucket,
    read_files_in_folder,
    delete_folder_from_bucket,
)
from functions_library import get_folder_paths
from pathlib import Path

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


# work-in-progress bucket
bucket_name = folder_dict["wip_bucket"]
# -

# ### Read files in bucket <a name="read"></a>
folder_name = models
read_files_in_folder(bucket_name, folder_name)

# ### Import individual files <a name="importfiles"></a>

# #### Importing individual models and associated files.
#
# >To import a specific model run - identified by `run_id`

run_id = "ramp_1_np_18_10_2023"

for folder in folders:
    destination_folder = folder_dict[folder]
    download_run_from_bucket(bucket_name, folder, destination_folder, run_id)

# #### To download just one file from one folder

# +
folder_name = models / "ramp_1_np_18_10_2023"
destination_folder = models
file_name = "ramp_1_np_18_10_2023"

download_run_from_bucket(bucket_name, folder_name, destination_folder, file_name)
# -

# ### Importing whole folders <a name="importfolders"></a>
#

# #### Importing all models and associated folders
#
# > Imports everything in `img`, `mask`, `models`, `outputs` folders

for folder in folders:
    destination_folder = folder_dict[folder]
    download_folder_from_bucket(bucket_name, folder_name, destination_folder)

# #### Importing a single folder

# +
folder_name = "ramp_bentiu_south_sudan/img"
destination_folder = ramp_img

download_folder_from_bucket(bucket_name, folder_name, destination_folder)
# -

# ### Delete folder from bucket <a name="delete"></a>

folder_name = "ramp_bentiu_south_sudan"
delete_folder_from_bucket(bucket_name, folder_name)
