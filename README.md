<img src="https://github.com/datasciencecampus/awesome-campus/blob/master/ons_dsc_logo.png">

![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)

# Somalia UNFPA Census Support

## Description

Automating building detection in satellite imagery over Somalia, with a focus on Internally displaced people (IDP) camps.

The first steps in this project is looking at the feasibility of applying the U-Net architecture to Planet SkySat Very-High-Resolution (VHR) satellite imagery. The U-Net model aims to detect formal and in-formal building structures to a high accuracy (>0.9). The feasibility study is focused on 5 areas in Somalia, with known IDP camps:

* Baidoa
* Beledweyne
* Kismayo
* Mogadishu

These areas were chosen due to being the focus of a recent Somalia National Bureau of Statistics (SNBS) study that surveyed building numbers and populations across IDP camps in the regions. The hope is that this study will provide some opportunity to ground-truth model outputs.

## Workflow

_in progress_

```mermaid
flowchart LR
    imagery[(planet<br>imagery)]-->qgis{QGIS}
    unfpa[(UNFPA<br>annotations)] -->qgis
    qgis-->|polygon<br>mask|sharepoint{<a href='https://officenationalstatistics.sharepoint.com/:f:/r/sites/dscdsc/Pro/2.%20Squads/International_Development/Data%20Science%20Projects/2.%20Data%20Science%20Research%20Projects/Somalia_UNFPA_census_support/Data/GCP%20ingress%20folder?csf=1&web=1&e=Pv6Icv'>SharePoint<br>GCP<br>ingest<br>folder</a>}
    qgis-->|image<br>raster|sharepoint
    sharepoint-->|img<br>file|preingress{preingress<br>notebook}
    sharepoint-->|mask<br>file|preingress
    preingress-->|checked<br>img file|sharepoint
    preingress-->|checked<br>mask file|sharepoint
    sharepoint-->|img<br>file|ingress{GCP<br>ingress<br>area}
    sharepoint-->|mask<br>file|ingress
    ingress-->|mask|processing[/training<br>data<br>processing<br>notebook\]
    ingress-->|raster|processing
    processing-->|numpy<br>arrays|train[/model<br>train<br>notebook\]
    train-->|numpy<br>arrays|results[/model<br>results<br>exploration<br>notebook\]
```
## Getting set-up (GCP):

This project is being developed in Google Cloud Platform (GCP), and so instructions will be specific to this environment. A determined user can hopefully generalise these across other tools.

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

### A note on Notebooks and Jupytext
notebooks in this project are stored as `.py` files with a hookup via Jupytext, to ensure proper version control. The notebooks are distinguishable from modular python scripts via the following comments at their beginning:
```
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
....
```
After cloning the repository, from your terminal run:
```
jupytext --to notebook <file_name>.py
```
 This will render a `.ipynb` file from the `.py` file. These two files are then synched together, such that any changes made to one will automatically update the other. This allows you to work and develop in a notebook, while avoiding the challenges and security threats that notebooks introduce in version control in terms of tracking changes and commiting outputs.


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

## Project structure tree
Successful running of the scripts assumes a certain structure in how where data and other auxiliary inputs need to be located.
The below tree demonstrates where each file/folder needs to be for successful execution or where files will be located following execution.

```
📦somalia_unfpa_census_support
 ┣ 📂data
 ┃ ┣ 📂training_data
 ┃ ┃ ┗ 📂img
 ┃ ┃ ┃ ┣ 📜training_data_<area>_<initial>.tif
 ┃ ┃ ┃ ┣ 📜training_data_<area>_<initial>.npy
 ┃ ┃ ┗ 📂mask
 ┃ ┃ ┃ ┣ 📜training_data_<area>_<initial>.shp
 ┃ ┃ ┃ ┣ 📜training_data_<area>_<initial>.npy
 ┣ 📂src
 ┃ ┣ 📜explore_imagery_and_data.py
 ┃ ┣ 📜functions_library.py
 ┃ ┣ 📜geospatial_util_functions.py
 ┃ ┣ 📜modelling_preprocessing.py
 ┃ ┣ 📜preingress_notebook.py
 ┃ ┣ 📜planet_img_processing_functions.py
 ┃ ┗ 📜model_train_notebook.py
 ┣ 📜.gitignore
 ┗ 📜README.md

```

## Training data

The training data only needs to be processed and outputted when first derived, or if changes are made to the polygons/raster. Follow the wiki guide to create training data and export as `.shp` files - using project naming structure:

`training_data_<area>_<your initials>`

## Before ingesting data onto GCP

Run the src/preingress_notebook.py prior to ingesting any data onto GCP to ensure the training data has been formatted correctly. 

## Things of note
The [wiki page attached to this repo](https://github.com/datasciencecampus/somalia_unfpa_census_support/wiki/Somalia-UNFPA-Census-support) contains useful resources and other relevant notes.
