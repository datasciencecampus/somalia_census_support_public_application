# Somalia UNFPA Census Support

## Description

Automating building detection in satellite imagery over Somalia, with a focus on Internally displaced people (IDPs).

## Getting set-up:

### Notebooks and Jupytext
For the benefit of proper version control, any notebooks in this project are stored as `.py` files with a hookup via Jupytext. The notebooks are distinguishable from modular python scripts via the following comments at their beginning:
```
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
....
```
In order to successfully use these as notebooks, you are required to have Jupytext [installed](https://jupytext.readthedocs.io/en/latest/install.html) (which can be achieved via a pip or conda install). Then, after cloning the repository, run 
```
jupytext --to notebook <file_name>.py
```
from your terminal. This will render a `.ipynb` file from the `.py` file. These two files are then synched together, such that any changes made to one will automatically update the other. This allows you to work and develop in a notebook, while avoiding the challenges and security threats that notebooks introduce in version control in terms of tracking changes and commiting outputs. 


## Workflow
Example, to be amended as required.

```mermaid
flowchart TD;
    A[step 1] --> B[step 2];
    C[additional file] --> B;
    B --> D[other stuff]
```

## Project structure tree
Successful running of the scripts assumes a certain structure in how where data and other auxiliary inputs need to be located.
The below tree demonstrates where each file/folder needs to be for successful execution or where files will be located following execution.

```
📦somalia_unfpa_census_support
 ┣ 📂data
 ┃ ┣ 📂IDP Priority Area Extent Shapefiles
 ┃ ┃ ┗ 📂IDP Priority Area Extent Shapefiles
 ┃ ┃ ┃ ┗ 📂IDP Survey Shapefiles
 ┃ ┃ ┃ ┃ ┣ 📜<areas>_Extent.cpg
 ┃ ┃ ┃ ┃ ┣ 📜<areas>_Extent.dbf
 ┃ ┃ ┃ ┃ ┣ 📜<areas>_Extent.prj
 ┃ ┃ ┃ ┃ ┣ 📜<areas>_Extent.shp
 ┃ ┃ ┃ ┃ ┣ 📜<areas>_Extent.shx
 ┃ ┣ 📂planet_images
 ┃ ┃ ┣ 📂Baidoa
 ┃ ┃ ┃ ┣ 📂Baidoa_NE_61MB_20220324
 ┃ ┃ ┃ ┣ 📂Baidoa_N_43MB_20221125
 ┃ ┃ ┃ ┣ 📂Baidoa_SW_107MB_20221125
 ┃ ┃ ┃ ┣ 📂Baidoa_S_66MB_20220324
 ┃ ┃ ┃ ┣ 📜Baidoa_NE_61MB_20220324.zip
 ┃ ┃ ┃ ┣ 📜Baidoa_N_43MB_20221125.zip
 ┃ ┃ ┃ ┣ 📜Baidoa_SW_107MB_20221125.zip
 ┃ ┃ ┃ ┗ 📜Baidoa_S_66MB_20220324.zip
 ┃ ┃ ┣ 📂Beletweyne
 ┃ ┃ ┃ ┣ 📂Beletweyne_45MB_20220818
 ┃ ┃ ┃ ┗ 📜Beletweyne_45MB_20220818.zip
 ┃ ┃ ┗ 📂Doolow
 ┃ ┃ ┃ ┣ 📂Dolow_E_95MB_20220830
 ┃ ┃ ┃ ┣ 📂Doolow_W_50MB_20221101
 ┃ ┃ ┃ ┣ 📜Dolow_E_95MB_20220830.zip
 ┃ ┃ ┃ ┣ 📜Doolow_W_50MB_20221101.zip
 ┃ ┣ 📂priority_areas_geojson
 ┃ ┃ ┣ 📜<area>_Extent.geojson
 ┃ ┣ 📂sentinel2_images
 ┃ ┃ ┣ 📂Doolow_S2
 ┃ ┃ ┃ ┗ 📂Doolow
 ┃ ┃ ┃ ┃ ┣ 📜Doolow_Extent_sentinel2_2022-08-15_2022-09-15.tif
 ┃ ┃ ┃ ┃ ┗ 📜Doolow_Extent_sentinel2_2022-10-15_2022-11-15.tif
 ┃ ┃ ┗ 📜Doolow_S2.zip
 ┃ ┗ 📜training_data.geojson
 ┣ 📂src
 ┃ ┣ 📜explore_imagery_and_data.py
 ┃ ┣ 📜functions_library.py
 ┃ ┣ 📜geospatial_util_functions.py
 ┃ ┣ 📜modelling_preprocessing.py
 ┃ ┣ 📜planet_img_processing_functions.py
 ┃ ┗ 📜sentinel_export_gee.py
 ┣ 📜.gitignore
 ┗ 📜README.md
 
```


### Obtaining Sentinel-2 images
Within the `src` folder there is a Python script for extracting Sentinel-2 imagery using Google Earth Engine. To execute this script, run 
```
python sentinel_export_gee.py <insert tags here for optional arguments>
```
For help on these optional arguments run 
```
python sentinel_export_gee.py -h
``` 
and see the detailed guidance on the [Uganda Forestry README](https://github.com/datasciencecampus/uganda_forestry/blob/master/acquire_sentinel2_imgs_readme.md#acquire-a-sentinel-2-image-using-google-earth-engine). 


## Things of note
The [wiki page attached to this repo](https://github.com/datasciencecampus/somalia_unfpa_census_support/wiki/Somalia-UNFPA-Census-support) contains useful resources and other relevant notes.
