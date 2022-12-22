# Somalia UNFPA Census Support

## Description

Automating building detection in satellite imagery over Somalia, with a focus on Internally displaced people (IDPs).

## Getting set-up:

### Sentinel-2 images
Within the `src` folder there is a Python script for extracting Sentinel-2 imagery using Google Earth Engine. To execute this script, run 
```
python sentinel_export_gee.py <insert tags here for optional arguments>
```
For help on these optional arguments run 
```
python sentinel_export_gee.py -h
``` 
and see the detailed guidance on the [Uganda Forestry README](https://github.com/datasciencecampus/uganda_forestry/blob/master/acquire_sentinel2_imgs_readme.md#acquire-a-sentinel-2-image-using-google-earth-engine). 

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
The below tree demonstrates where each file/folder needs to be for successful execution.

```
📦somalia_unfpa_census_support
 ┣ 📂data
 ┣ 📂src
 ┣ 📜.gitignore
 ┗ 📜README.md
```

## Things of note
The [wiki page attached to this repo](https://github.com/datasciencecampus/somalia_unfpa_census_support/wiki/Somalia-UNFPA-Census-support) contains useful resources and other relevant notes.
