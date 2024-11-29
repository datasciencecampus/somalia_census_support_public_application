# Cloud Platform README

## Overview
This README contains functionality for using cloud platforms (specifically Google Cloud Platform) to host the training pipeline.

As an example of how to set up your data storage to work with our functionality, our project was set up as follows:

```
📦ingress-bucket
 ┣ 📂area_name
 ┃	┣ 📜image_1_shapefile.tif
 ┃	┣ 📜image_1_polgyons.tif

📦wip-bucket
 ┣ 📂models
 ┃	┣ 📜model_name_date.h5
 ┣ 📂outputs
 ┃	┣ 📜model_name_date_conditions.txt
 ┃	┣ 📜model_name_date_filenames.npy
 ┃	┣ 📜model_name_date_xtest.npy
 ┃	┣ 📜model_name_date_ytest.npy
 ┃	┣ 📜model_name_date_ypred.npy

📦egress-bucket
 ┣ 📂conditions
 ┃	┣ 📜model_name_date_conditions.txt
 ┣ 📂footprints
 ┃	┣ 📜image_1_footprints.geojson
 ┣ 📂models
 ┃	┣ 📜model_name_date.h5
 ┣ 📂outputs
 ┃	┣ 📜model_name_date_conditions.txt
 ┃	┣ 📜model_name_date_filenames.npy
 ┃	┣ 📜model_name_date_xtest.npy
 ┃	┣ 📜model_name_date_ytest.npy
 ┃	┣ 📜model_name_date_ypred.npy
```

### Code
```
📦somalia_unfpa_census_support
 ┣ 📂src
 ┃ ┗ 📂cloud
 ┃    ┣ 📜bucket_access_functions.py
 ┃    ┣ 📜bucket_export_notebook.py
 ┃    ┣ 📜bucket_import_notebook.py
 ┃    ┣ 📜download_from_bucket.py
 ┃    ┣ 📜upload_to_bucket.py
 ┃    ┗ 📜cloud_functionality_readme.py
```
