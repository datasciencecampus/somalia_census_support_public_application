# ## Notebook for exporting data to WIP bucket
#
# > More functions including how to delete files are in `bucket_access_functions.py`

from pathlib import Path
from bucket_access_functions import move_file_to_bucket
from functions_library import get_folder_paths

# +
folder_dict = get_folder_paths()

# Set directories to pull run files from
model_dir = Path(folder_dict["models_dir"])
output_dir = Path(folder_dict["outputs_dir"])

# work-in-progress bucket
bucket_name = folder_dict["wip_bucket"]
# -

run_id = "outputs_alt_test"

# ### Upload the model to the WIP bucket

for file in model_dir.iterdir():
    if file.name.startswith(run_id):
        move_file_to_bucket(file, bucket_name)

# ### Upload the model outputs to the WIP bucket

for file in output_dir.iterdir():
    if file.name.startswith(run_id):
        move_file_to_bucket(file, bucket_name)

# ### Upload the additional config files to the WIP bucket

for file in output_dir.iterdir():
    if file.name.startswith(run_id):
        move_file_to_bucket(file, bucket_name)
