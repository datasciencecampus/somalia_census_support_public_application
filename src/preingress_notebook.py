# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pre GCP Ingress Notebook
#
# > Notebook to be run before any files are transferred to the SharePoint GCP ingress folder
#
# > You will need to copy contents of GCP_ingress folder on Sharepoint to your local machine

# %% [markdown]
# ## Set-up

# %%
# Load required libraries
from pathlib import Path # working with file paths
import geopandas as gpd # working with geospatial files and data
import warnings # used for sending warnings
import re # pattern matching

# Local imports
from functions_library import setup_sub_dir

# %%
# Note directories of interest
data_dir = Path.cwd().parent.joinpath("data")
training_data_dir = data_dir.joinpath("training_data")
img_dir = setup_sub_dir(training_data_dir, "img") # Note setup_sub_dir creates these if not present
mask_dir = setup_sub_dir(training_data_dir, "mask")

# %% [markdown]
# ## Explore files

# %%
# Get all the img and mask files present
img_files = list(img_dir.glob("*.tif"))
mask_files = list(mask_dir.glob("*.geojson"))

# Check that same number of imgs and mask files present
if len(img_files) != len(mask_files):
    warnings.warn(f"Number of image files {len(img_files)} doesn't match number of mask files {len(mask_files)}")

# %% [markdown]
# ##### List img files in img folder

# %%
img_file_names = [file.name for file in img_files]
img_file_names

# %% [markdown]
# ##### List mask files in mask folder

# %%
mask_file_names = [file.name for file in mask_files]
mask_file_names


# %% [markdown]
# ## General file cleaning
#
# * change all file names to lower case (see [`Path.rename()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename) and [`str.lower()`](https://www.programiz.com/python-programming/methods/string/lower)
# * check there each img file has corresponding mask file and _vice versa_ - both img and mask files should have same name except suffix
# * ensure naming convention upheld for tif and geojson? Should be: `training_data_<area>_<tile no>_<initials>_<bgr>.tif`
# * specific for img files! check banding? Check in with Laurence on this and see: https://github.com/datasciencecampus/somalia_unfpa_census_support/issues/173

# %% [markdown]
# ##### Change all file names to lower case

# %%
def change_to_lower_case(files):
    
    # Makes img and mask file names lower case and stores in list
    files_lower = [Path(file.parent, file.name.lower()) for file in files]
    
    # Loop through both lists to rename original img files name
    for file_original, file_lower in zip(files, files_lower):
        file_original.rename(file_lower)
        
    return files_lower


# %%
img_files_lower = change_to_lower_case(img_files)
mask_files_lower = change_to_lower_case(mask_files)


# %% [markdown]
# ###### Check each mask file has corresponding img file

# %%
def check_mask_file_for_img_file(img_files_lower, mask_files_lower):
    
    # get mask file names
    mask_file_names = [mask_file.name for mask_file in mask_files_lower]

    # examine each img file
    for img_file in img_files_lower:
        
        # initialise mask name variable
        mask_file_name = None
        
        # check if banding present
        if "bgr" in img_file.name or "rgb" in img_file.name:
            mask_file_name = img_file.name[:-8] + ".geojson"
        
        else:
            mask_file_name = img_file.name[:-4] + ".geojson"
            warnings.warn(f"banding pattern isn't present in {img_file.name}")
        
        # check if mask file present
        if not mask_file_name in mask_file_names:
            warnings.warn(f"The mask file ({mask_file_name}) for img_file ({img_file.name}) doesn't exist")
    


# %%
check_mask_file_for_img_file(img_files_lower, mask_files_lower)


# %% [markdown]
# ###### Check each img file has corresponding mask file 

# %%
def check_img_file_for_mask_file(img_files_lower, mask_files_lower):
    
    # get img file names
    img_file_names = [img_file.name for img_file in img_files_lower]

    # examine each mask file
    for mask_file in mask_files_lower:
        
        # build img file name
        img_file_name = mask_file.name[:-8] + "_bgr.tif"
        
        # check if img file present
        if not img_file_name in img_file_names:
            warnings.warn(f"The img file ({img_file_name}) for mask_file ({mask_file.name}) doesn't exist")


# %%
check_img_file_for_mask_file(img_files_lower, mask_files_lower)


# %% [markdown]
# ##### Check to ensure naming convention held for masks and img files

# %%
def check_naming_convention_upheld(img_files_lower, mask_files_lower):
    
    for file in img_files_lower + mask_files_lower:
    
        # checks if naming convention correct for mask and img files
        if not re.match(r"training_data_.+_[0-9]+_*", file.name):
            warnings.warn(f"The naming convention for ({file.name}) is not correct. Please change")


# %%
check_naming_convention_upheld(img_files_lower, mask_files_lower)

# %% [markdown]
# ## Mask file cleaning
#
# * check data in each geojson (see [reading geojson](https://docs.astraea.earth/hc/en-us/articles/360043919911-Read-a-GeoJSON-File-into-a-GeoPandas-DataFrame)):
#    * check there is a type column
#    * remove fid or id column
#    * check for na

# %%
# Read multiple geojsons files into separate geopandas DataFrames

# assign dataset names from list
mask_files

# append geojsons into the list
for mask_file in mask_files_lower:
    temp_df = gpd.read_file(str(mask_file))
    
    # drop fid and id columns
    
    # check for type column - if not send error
    
        # check any null values in type - send error
      
    # check for na values - send error or warning
    
    # write back to geojson



# %%
temp_df = gpd.read_file(str(mask_files_lower[0]))

# %%
temp_df

# %% [markdown]
# ### Start data cleaning

# %%

# %%
# Drop "fid" column
for i in :
    
    if "fid" in dataframes_list:
        dataset = dataset.drop(columns=["fid"])
    else:
        dataset = dataset.drop(columns=["id"])

# %%
# If statement with warning to see if "Type" column has missing values

    if dataframes_list["Type"].isnull().values.any():
        warnings.warn(f"Type has null values")
    else: 
        print("No null values")

# %%
dataframes_list[2].head()

# %%
training_data_baidoa_1_JO = gpd.read_file(mask_dir.joinpath("training_data_baidoa_1_JO.geojson"))
# training_data_baidoa_1_JO.head()

# %%
# Check there is a type column and to see whether any missing values

training_data_baidoa_1_JO.info()

# %%
# Drop "fid" column

if "fid" in training_data_baidoa_1_JO:
    training_data_baidoa_1_JO = training_data_baidoa_1_JO.drop(columns=["fid"])
else:
    training_data_baidoa_1_JO = training_data_baidoa_1_JO.drop(columns=["id"])

# %%
training_data_baidoa_1_JO.isnull().values.any()

# %%
# If statement with warning to see if "Type" column has missing values

if training_data_baidoa_1_JO["Type"].isnull().values.any():
    warnings.warn(f"Type has null values")
else: 
    print("No null values")

# %%
