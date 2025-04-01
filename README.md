<img src="https://github.com/datasciencecampus/awesome-campus/blob/master/ons_dsc_logo.png">

![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)

# Somalia UNFPA Census Support - Training Development Branch

## Disclaimer

This branch has been uploaded to provide the training structure that we used to train our U-Net model for the public application. The ONS Data Science Campus place no guarantee on the efficacy of any other models trained using this method.


## How to use this repo

This repository has three primary functions:
1. train a multi-class UNet model
2. evaluate model outputs
3. create footprints from unseen satellite images

The workflow was built and optimised using Planet SkySat 0.5m/pixel imagery, with training tiles 384 x 384 pixels in size and utilising a 4th NIR channel. Any VHR satellite imagery 384 x 384 px (or larger) should be able to be used in this project, although this has not been tested.

Training data consists of `geoTIF` image rasters with corresponding `geoJSON` masks of polygons. The masks have been manually created for this project. Both inputs are ~384 x 384 in size.

Code is written in python with the intention of being run in Jupyter notebooks.

The work was developed in Google Cloud Platform (GCP) infrastructure using a NVIDI T4 X 1 notebook with 16 vCPUs, 104 GB RAM, and 1 GPU.

GCP specific notebooks are highlighted below. All packages are provided in a `requirements.txt`, for a dedicated user to generalise this workflow in other environments.


## Workflow

Creation of Training Data
```mermaid
flowchart LR
    imagery[(planet<br>imagery)]-->qgis{QGIS}
    unfpa[(UNFPA<br>annotations)] -->qgis
    qgis-->|polygon<br>mask|preingress[/preingress<br>notebook\]
    qgis-->|image<br>raster|preingress
    preingress-->|checked<br>mask|sharepoint
    preingress-->|checked<br>img|sharepoint{<a href='https://officenationalstatistics.sharepoint.com/:f:/r/sites/dscdsc/Pro/2.%20Squads/International_Development/Data%20Science%20Projects/2.%20Data%20Science%20Research%20Projects/Somalia_UNFPA_census_support/Data/GCP%20ingress%20folder?csf=1&web=1&e=Pv6Icv'>SharePoint<br>GCP<br>ingest<br>folder</a>}
    sharepoint-->|mask<br>file|ingress{GCP<br>ingress<br>area}
    sharepoint-->|img<br>file|ingress
```
Model training
```mermaid
flowchart LR
    ingress{GCP<br>ingress<br>area}-->download[/download data<br>from ingress<br>notebook\]
    download-->|mask file|local
    download-->|img file|local
    local{Local<br>GCP<br>Env.}-->|mask file|processing[/pre-modelling<br>notebook\]
    local{Local<br>GCP<br>Env.}-->|img file|processing[/pre-modelling<br>notebook\]
    processing-->|numpy<br>arrays|dataaug[/data<br>augmentation<br>notebook\]
    dataaug-->|numpy<br>arrays|train[/model<br>train<br>notebook\]
    train-->|numpy arrays|outputs
    train-->|history|outputs
    train-->|hdf5|outputs
    outputs[model<br>outputs<br>notebook]

```

## Getting set-up (GCP):

Users should clone the repo within their personal GCP notebooks, which are accessed via the Vertex AI Workbench.


### Virtual environment
Once in the project space (i.e. the base repository level) it is recommended you set-up a virtual environment. In the terminal run:
```
python3 -m venv venv-somalia-gcp
```
Next, to activate your virtual environment run
```
source venv-somalia-gcp/bin/activate
```

### Install dependencies
While in your active virtual environment, perform a pip install of the `requirements.txt` file, which lists the required dependencies. To do this run:
```
pip install -r requirements.txt
```

### Set-up custom kernel from your virtual environment
To access your installed packages from your virtual environment you need to set-up an ipython kernel from your environment. By default, the notebooks in GCP will access the base python. To set-up a custom kernel, ensure your virtual enivronment is active and from the terminal run:
```
ipython kernel install --name "venv-somalia-gcp" --user
```

After some possible delay, the kernel should appear in the list of kernels available in the top right corner of your notebooks.


### Pre-commit actions
This repository makes use of [pre-commit hooks](https://towardsdatascience.com/getting-started-with-python-pre-commit-hooks-28be2b2d09d5). If approaching this project as a developer, you can install and enable `pre-commit` by running the following in your shell:
   1. Install `pre-commit`: within your active virtual/conda environment, run

      ```
      pip install pre-commit
      ```
   2. Enable `pre-commit`: Ensure you at the base repository level and run

      ```
      pre-commit install
      ```
Once pre-commits are activated, whenever you commit to this repository a series of checks will be excuted. The pre-commits include checking for security keys, large files, unresolved merge conflict headers and will also automatically format the code to an agreed standard. The use of active pre-commits are highly encouraged when working with this codebase.

*NOTE:* When a pre-commit hook fails, it will often automatically make modifications to the files you are attempting to commit. However, the pre-commit set-up will not be able to correct all errors itself, so take note of any flagged issues and resolve these manually. In either event, the commit will not yet have been confirmed. You will be required to perform a `git add` and then redo the `git commit` in order to proceed (such as pushing to origin).

### A note on Jupyter notebooks and Jupytext
Notebooks in this project are stored as `.py` files with a hookup via Jupytext, to ensure proper version control. The notebooks are distinguishable from modular python scripts via the following comments at their beginning:
```
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
```
After cloning the repository, from your terminal run:
```
jupytext --to notebook <file_name>.py
```
This will render a `.ipynb` file from the `.py` file. These two files are then synched together, such that any changes made to one will automatically update the other. This allows you to work and develop in a notebook, while avoiding the challenges and security threats that notebooks introduce in version control in terms of tracking changes and commiting outputs.

Note ensure ` jupytext_version: 1.14.5` for syncing across the project.

## Project structure tree
Successful running of the scripts assumes a certain structure in how where data and other auxiliary inputs need to be located.
The below tree demonstrates where each file/folder needs to be for successful execution or where files will be located following execution.

### Overview
```
📦somalia_unfpa_census_support
 ┣ 📂data
 ┣ 📂models
 ┣ 📂outputs
 ┣ 📂src
 ┣ 📂venv-somalia-gcp
 ┣ 📜config.yaml
 ┣ 📜.gitignore
 ┣ 📜requirements.text
 ┗ 📜README.md

```
### Data
```
📦somalia_unfpa_census_support
 ┣ 📂data
 ┃ ┣ 📂camp_tiles
 ┃ ┃ ┗ 📂baidoa
 ┃ ┣ 📂footprints
 ┃ ┃ ┗ 📜<area>_<sub_area>.geojson
 ┃ ┣ 📂outputs
 ┃ ┣ 📂training
 ┃ ┃ ┣ 📂json_dir
 ┃ ┃ ┃  ┗ 📜t<data_type>_features_dict.json
 ┃ ┃ ┣ 📂training_data
 ┃ ┃ ┃  ┣ 📂img
 ┃ ┃ ┃  ┃   ┣ 📜training_data_<area>_<initial>.tif
 ┃ ┃ ┃  ┃   ┗ 📜training_data_<area>_<initial>.npy
 ┃ ┃ ┃  ┣ 📂mask
 ┃ ┃ ┃  ┃   ┣ 📜training_data_<area>_<initial>.geojson
 ┃ ┃ ┃  ┃   ┗ 📜training_data_<area>_<initial>.npy
 ┃ ┃ ┣ 📂validation_data
 ┃ ┃ ┃  ┣ 📂img
 ┃ ┃ ┃  ┃   ┣ 📜validation_data_<area>_<initial>.tif
 ┃ ┃ ┃  ┃   ┗ 📜validation_data_<area>_<initial>.npy
 ┃ ┃ ┃  ┣ 📂mask
 ┃ ┃ ┃  ┃   ┣ 📜validation_data_<area>_<initial>.geojson
 ┃ ┃ ┃  ┃   ┗ 📜validation_data_<area>_<initial>.npy
 ┃ ┃ ┣ 📂ramp_bentiu_south_sudan
 ┃ ┃ ┃  ┣ 📂img
 ┃ ┃ ┃  ┃   ┣ 📜ramp_bentiu_south_sudan_<area>_<initial>.tif
 ┃ ┃ ┃  ┃   ┗ 📜vramp_bentiu_south_sudan_<area>_<initial>.npy
 ┃ ┃ ┃  ┣ 📂mask
 ┃ ┃ ┃  ┃   ┣ 📜ramp_bentiu_south_sudan_<area>_<initial>.geojson
 ┃ ┃ ┃  ┃   ┗ 📜ramp_bentiu_south_sudan_<area>_<initial>.npy
 ┃ ┃ ┣ 📂stacked_arrays
 ┃ ┃ ┃  ┣ 📂img
 ┃ ┃ ┃  ┃   ┣ 📜<data_type>_all_stacked_images.npy
 ┃ ┃ ┃  ┃   ┗ 📜<data_type>_all_stacked_filenames.npy
 ┃ ┃ ┃  ┣ 📂mask
 ┗ ┗ ┗  ┗   ┗ 📜<data_type>_all_stacked_masks.npy

```
### Code
```
📦somalia_unfpa_census_support
 ┣ 📂src
 ┃ ┣ 📜1_premodelling_notebook.py
 ┃ ┣ 📜2_data_augmentation_notebook.py
 ┃ ┣ 📜3_model_train_notebook.py
 ┃ ┣ 📜4_model_outputs_notebook.py
 ┃ ┣ 📜5_model_run_evaluation.py
 ┃ ┣ 📜bucket_access_functions.py
 ┃ ┣ 📜bucket_export_notebook.py
 ┃ ┣ 📜bucket_import_notebook.py
 ┃ ┣ 📜create_footprints.py
 ┃ ┣ 📜create_footprints_functions.py
 ┃ ┣ 📜create_input_tiles.py
 ┃ ┣ 📜data_augmentation_functions.py
 ┃ ┣ 📜download_from_bucket.py
 ┃ ┣ 📜functions_library.py
 ┃ ┣ 📜loss_functions.py
 ┃ ┣ 📜idp_map_notebook.py
 ┃ ┣ 📜image_processing_functions.py
 ┃ ┣ 📜mask_processing_functions.py
 ┃ ┣ 📜model_outputs_functions.py
 ┃ ┣ 📜multi_class_unet_model_build.py
 ┃ ┣ 📜preingress_notebook.py
 ┃ ┣ 📜weight_functions.py
 ┣ 📜config.yaml
 ┣ 📜.gitignore
 ┣ 📜requirements.text
 ┗ 📜README.md

```

## Data

### Training data

The original project used the following naming structure for it's training data:

`training_data_<area>_<unique int>_<your initials>`

>For validation data replace `training` with `validation`.

If you want to remove the check that verifys this pattern, disable `check_naming_convention_upheld` in the `preingress_notebook`.

### Uploading data to GCP

We have included the code for uploading/downloading data from a GCP environment. These functions may vary if using AWS/Azure/another provider.

Note the directory structure, which mirrors that of local GCP (shown above).

### Moving data to local GCP storage

To download files from the ingress bucket into the local GCP environment run the `download_from_bucket.py` script with the below line:

```
python src/download_from_bucket.py gs://<GCP-Instance>-ingress/training/ data/training/
```

or moving model files from the `wip` bucket to local storage:
```
python src/download_from_bucket.py gs://<GCP-Instance-wip>/models/ models/
```

### Moving data from local GCP storage

To upload files from local storage into the egress bucket run the `upload_to_bucket.py` script with the below line:
```
python src/upload_to_bucket.py data/outputs/figures gs://<GCP-Instance>-egress/
```
or moving model files into the `wip` bucket, it's the same script but different bucket location:
```
python src/upload_to_bucket.py models/ gs://<GCP-Instance>-wip/models
```

## Creating shelter footprints

In the scenario where you want to use pre-trained models to create building footprints, only the `create_footprints.py` script is required to be run. Pre-trained models are available in the `wip` bucket, these should be downloaded locally to run the notebook (see `Moving data from local GCP storage` section above). To run `create_footprints.py` `conditions.txt` for model runs and `camp_tiles` files also need to be downloaded. 

To get `conditions.txt` run:
```
python src/download_from_bucket.py gs://<GCP-Instance>-wip/outputs/ data/outputs/
```

To get `camp_tiles` for Baidoa for example, run:
```
python src/download_from_bucket.py gs://<GCP-Instance>-ingress/baidoa/ camp_tiles/baidoa/
```
