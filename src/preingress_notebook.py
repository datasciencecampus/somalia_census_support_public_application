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
# ### List img files in img folder

# %%
img_file_names = [file.name for file in img_files]
img_file_names

# %% [markdown]
# ### List mask files in mask folder

# %%
mask_file_names = [file.name for file in mask_files]
mask_file_names

# %% [markdown]
# ## General file cleaning
#
# * change all file names to lower case (see [`Path.rename()`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename) and [`str.lower()`](https://www.programiz.com/python-programming/methods/string/lower)
# * check there each img file has corresponding mask file and _vice versa_ - both img and mask files should have same name except suffix
# * ensure naming convention upheld? Should be: `training_data_<area>_<tile no>_<initials>_<bgr>.tif`
# * check banding? Check in with Laurence on this and see: https://github.com/datasciencecampus/somalia_unfpa_census_support/issues/173

# %%
img_files

# %%
# for loop to go through img names and convert to lower case
for i in range(len(img_files)):
    lower_case_img_file_names = img_files[i].name.lower()
    print(lower_case_img_file_names)

# %%
mask_files

# %%
# for loop to go through mask names and convert to lower case
for i in range(len(mask_files)):
    lower_case_mask_file_names = mask_files[i].name.lower()
    print(lower_case_mask_file_names)

# %%
# chop off file name in path lib
# convert file path to string
# combine string with lowercase name
# convert to path
# rename the file

# %%
str(mask_files[0])

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

# create empty list
dataframes_list = []

# append geojsons into the list
for i in range(len(mask_files)):
    temp_df = gpd.read_file(str(mask_files[i]))
    dataframes_list.append(temp_df)
    
# display geopandas dataframe
for dataset in dataframes_list:
    display(dataset)

# %%
dataframes_list[4]

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
