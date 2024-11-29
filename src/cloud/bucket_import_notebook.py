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
) = [folder_dict[folder] for folder in folders]


# work-in-progress bucket
bucket_name = folder_dict["wip_bucket"]
# -

# ### Read files in bucket <a name="read"></a>

folder_name = "models"
read_files_in_folder(bucket_name, folder_name)

# ### Import individual files <a name="importfiles"></a>

run_id = ""

# #### For Outputs (6 files)

folder_name = "outputs"
destination_folder = outputs
download_run_from_bucket(bucket_name, folder_name, destination_folder, run_id)

# #### For Models (1 file)

# +
folder_name = "models"
destination_folder = models

download_run_from_bucket(bucket_name, folder_name, destination_folder, run_id)
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
