# ## Notebook for exporting data to WiP bucket
#
# <div class="alert alert-block altert-danger">
#     <i class="fa fa-exclamation-triangle"></i> check the kernel in the above right is `python3` <b>not</b> `venv-somalia-gcp`
# </div>
#
# **Purpose**
#
# To export files or folders from local GCP storage to the WiP or review bucket.
#
# **Things to note**
#
# - Do not run all cells in this notebook, instead choose the function you want to perform (export a file or folder) and run only that action
#
# #### Jump to:
# 1. ##### [Read files in bucket](#read)
# 1. ##### [Delete files in bucket](#delete)
# 1. ##### [Exporting individual files](#exportfiles)
# 1. ##### [Exporting folders](#exportfolders)

# ### Set-up

from bucket_access_functions import (
    move_file_to_bucket,
    move_folder_to_bucket,
    delete_folder_from_bucket,
    read_files_in_bucket,
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


# buckets
wip_bucket = folder_dict["wip_bucket"]
review_bucket = folder_dict["review_bucket"]
# -

# ### Read files in bucket <a name="read"></a>

read_files_in_bucket(review_bucket)

# ### Delete folder from bucket <a name="delete"></a>

delete_folder_from_bucket(bucket_name=wip_bucket, folder_name="ramp_bentiu_south_sudan")

# ### Exporting individual files <a name="exportfiles"></a>

run_id = "qa_testing_2024-01-30_1655"

# #### Export model file to WiP bucket

for file in models.iterdir():
    if file.name.startswith(run_id):
        move_file_to_bucket(source_file_path=file, bucket_name=wip_bucket)

# #### Export model outputs to WiP bucket

for file in outputs.iterdir():
    if file.name.startswith(run_id):
        move_file_to_bucket(source_file_path=file, bucket_name=wip_bucket)

# ### Exporting folders <a name="exportfolders"></a>

move_folder_to_bucket(
    source_folder=models, bucket_name=wip_bucket, destination_folder_name="models"
)

# #### Export .shp file to review bucket

# example file for now
file = Path("../outputs/qa_testing_2024-01-31_0658_ytest.npy")
move_file_to_bucket(source_file_path=file, bucket_name=review_bucket)
