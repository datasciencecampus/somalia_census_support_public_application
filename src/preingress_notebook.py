# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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
# > You will need to manually copy contents of GCP_ingress folder on Sharepoint to your local machine and vice-versa

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

# Absolute path for img files
img_files = list(img_dir.glob("*.tif"))

# Absolute path for mask files
mask_files = list(mask_dir.glob("*.geojson"))

# Check that same number of imgs and mask files present - if not then warning
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
    
    """
    Changes file names for img and mask files to lower case in training data before ingress to GCP.
    
    Parameters
    ----------
    files: list
        Absolute paths for img or mask files on local machine
        
    Returns
    -------
    files_lower : list
        Lower case img and mask file names stored in list.
    """
    
    # Makes img and mask file names lower case and stored in list
    files_lower = [Path(file.parent, file.name.lower()) for file in files]
    
    # Loop through both lists to rename original img or mask file name
    for file_original, file_lower in zip(files, files_lower):
        file_original.rename(file_lower)
        
    return files_lower


# %%
# Lower case img file names
img_files_lower = change_to_lower_case(img_files)

# Lower case mask file names
mask_files_lower = change_to_lower_case(mask_files)


# %% [markdown]
# ###### Check each mask file has corresponding img file & check each img file has corresponding mask file 

# %%
def vice_versa_check_mask_file_for_img_file(img_files_lower, mask_files_lower, for_mask_or_img):

    """
        Checks mask file names to see if they have a corresponding img file in training data before ingress to GCP.
        Will also check if img file names to see if they have a corresponding mask file
    
        Parameters
        ----------
        img_files_lower: list
            List of img files in lower case
        mask_files_lower: list
           List of mask files in lower case 
        for_mask_or_img: character
           If "mask" then will check whether mask files have an img file with the same name. If "img" will do
           check whether img files have a mask file with the same name.
       
        Returns
        -------
        Warning if there is not a corresponding img file for mask file name or vice versa
        """
    
    if for_mask_or_img == "mask":
        print("Checking mask file names for corresponding img file")
    
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
        
            # check if mask file present - if not then warning
            if not mask_file_name in mask_file_names:
                warnings.warn(f"The mask file ({mask_file_name}) for img_file ({img_file.name}) doesn't exist")
    
    
    elif for_mask_or_img == "img":
            print("Checking img file names for corresponding mask file")
        
            # get img file names
            img_file_names = [img_file.name for img_file in img_files_lower]

            # examine each mask file
            for mask_file in mask_files_lower:
        
                # build img file name
                img_file_name = mask_file.name[:-8] + "_bgr.tif"
        
                # check if img file present - if not then warning
                if not img_file_name in img_file_names:
                    warnings.warn(f"The img file ({img_file_name}) for mask_file ({mask_file.name}) doesn't exist")

# %%
vice_versa_check_mask_file_for_img_file(img_files_lower, mask_files_lower, for_mask_or_img = "mask")

# %%
vice_versa_check_mask_file_for_img_file(img_files_lower, mask_files_lower, for_mask_or_img = "img")


# %% [markdown]
# ##### Check to ensure naming convention held for masks and img files

# %%
def check_naming_convention_upheld(img_files_lower, mask_files_lower):
    
     """
    Checks correct naming convention is being used for img and mask files in training data before ingress to GCP.
    
    Parameters
    ----------
    img_files_lower: list
        List of img files in lower case
    mask_files_lower: list
        List of mask files in lower case 
    
    Returns
    -------
    Warning if naming convention for mask or img file is incorrect and informs what to change to
    """
        
    for file in img_files_lower + mask_files_lower:
    
        # checks if naming convention correct for mask and img files
        if not re.match(r"training_data_.+_[0-9]+_*", file.name):
            warnings.warn(f"The naming convention for ({file.name}) is not correct. Please change to training_data_<area>_<tile no>_<initials>_<bgr>.tif for imgs or training_data_<area>_<tile no>_<initials>.geojson for masks ")


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
def cleaning_of_mask_files(mask_files_lower):
    
     """
    Cleans geopandas dataframes of all mask files and then overwrites them in the mask folder. Checks for
    additional columns, missing columns and null values.
    
    Parameters
    ----------
    mask_files_lower: list
        List of mask files in lower case 
    
    Returns
    -------
    Warning message if "Type" column not found along with creation of new column in geopandas dataframe, 
    print message and lastly a GeoJSON file that has been cleaned. 
    """
        
    for mask_file in mask_files_lower:
        temp_df = gpd.read_file(str(mask_file))
    
        # drop fid and id columns
     
        if "id" in temp_df:
            temp_df = temp_df.drop(columns=["id"])
        
        elif "fid" in temp_df:
            temp_df = temp_df.drop(columns=["fid"])
        
        # check for type column - if not send error
    
        if "Type" in temp_df.columns:
            print(f"Type column is present for {(mask_file.name)}")
        
        else:
            temp_df["Type"] = 0
            warnings.warn(f"The Type column not found {(mask_file.name)} setting to background")
    
        # check any null values in type column - send error
    
        if temp_df["Type"].isnull().values.any():
            warnings.warn(f"Type column for ({mask_file.name}) has null values")
        
        else: 
            print(f"No null values present in Type column for ({mask_file.name})")
        
        # write back to geojson
    
        temp_df.to_file(mask_dir.joinpath(f"{(mask_file)}"), driver="GeoJSON")


# %%
cleaning_of_mask_files(mask_files_lower)
