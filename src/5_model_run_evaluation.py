# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: venv-somalia-gcp
#     language: python
#     name: venv-somalia-gcp
# ---

# %% [markdown]
# # Model Run Evaluation
#
# <div style="padding: 15px; border: 1px solid transparent; border-color: transparent; margin-bottom: 20px; border-radius: 4px; color: #31708f; background-color: #d9edf7; border-color: #bce8f1;">
# Before running this project ensure that the correct kernel is selected (top right). The default project environment name is `venv-somalia-gcp`.
# </div>
#
# This notebook evaluates model runs to allow for comparisons. Can see how individual tiles performed.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions_library import get_folder_paths
from pathlib import Path

# %%
folder_dict = get_folder_paths()
# set model and output directories
outputs_dir = Path(folder_dict["outputs_dir"])
model_dir = Path(folder_dict["models_dir"])

# %% [markdown]
# ### Model Run Conditions

# %%
runid = "qa_testing_2024-01-31_0658"

# %% [markdown]
# #### Load run conditions

# %%
# load in model run conditions txt file
model_run_conditions_file = f"{runid}_conditions.txt"

model_run_conditions = outputs_dir / model_run_conditions_file

run_conditions = open(model_run_conditions, "r")

print(run_conditions.read())

# %% [markdown]
# #### Save run conditions

# %%
# convert run conditions to dataframe
model_run_conditions_df = pd.read_csv(model_run_conditions, header=None, names=[runid])

model_run_conditions_csv_filename = runid + "_conditions.csv"

model_run_conditions_csv_file_path = outputs_dir / model_run_conditions_csv_filename

# save run conditions to csv
model_run_conditions_df.to_csv(model_run_conditions_csv_file_path, index=None)

# %% [markdown]
# ### Building Counts

# %%
# create df of csv building counts
file_pattern = "*_building_polygon_stats.csv"
csv_files = outputs_dir.glob(file_pattern)

dataframes = []

for csv_file in csv_files:
    building_polygon_stats_df = pd.read_csv(
        csv_file,
        usecols=[
            "filename",
            "tent_actual",
            "building_actual",
            "accuracy_percentage_tent",
            "accuracy_percentage_building",
        ],
    )
    # adding area name as column
    building_polygon_stats_df["area"] = (
        building_polygon_stats_df["filename"].str.split("_").str[2]
    )
    # adding csv file name as column
    building_polygon_stats_df["csv_name"] = csv_file.stem
    dataframes.append(building_polygon_stats_df)


merged_building_polygon_stats_df = pd.concat(dataframes, ignore_index=True)

# removing background tiles
merged_building_polygon_stats_df = merged_building_polygon_stats_df[
    ~merged_building_polygon_stats_df["filename"].str.endswith("_background")
]

merged_building_polygon_stats_df

# %%
##########

# %%
# find rows where 'accuracy_percentage_tent' is -inf
inf_rows_merged_building_polygon_stats_df = merged_building_polygon_stats_df[
    merged_building_polygon_stats_df["accuracy_percentage_tent"] == -np.inf
]

# remove rows where 'accuracy_percentage_tent' is -inf
merged_clean_building_polygon_stats_df = merged_building_polygon_stats_df[
    merged_building_polygon_stats_df["accuracy_percentage_tent"] != -np.inf
]

merged_clean_building_polygon_stats_df

# %%
####### find out where inf is coming from

# %% [markdown]
# #### Tent Stats

# %%
# group df by csv name, aggregate stats for tents and reset index
tent_stats_grouped_df = (
    merged_clean_building_polygon_stats_df.groupby("csv_name")[
        "accuracy_percentage_tent"
    ]
    .agg(["min", "max", "mean"])
    .reset_index()
)

# rename columns
tent_stats_grouped_df.columns = [
    "csv_name",
    "min_accuracy_tent",
    "max_accuracy_tent",
    "avg_accuracy_tent",
]

tent_stats_grouped_df


# %%
#### same as merged_clean_building_polygon_stats_df so not sure if needed?

# %%
exclude_filenames = [
    "border_testing_2023-12-20_1108_building_polygon_stats",
    "border_testing_2024-01-15_1733_building_polygon_stats",
    "border_testing_2024-01-15_1843_building_polygon_stats",
]

# removing low building count csvs
building_polygon_stats_filtered_df = merged_clean_building_polygon_stats_df[
    ~merged_clean_building_polygon_stats_df["csv_name"].isin(exclude_filenames)
]

building_polygon_stats_filtered_df

# %%
###########################

# %% [markdown]
# ### Tent accuracy percentage

# %%
# make copy of filtered_df
tent_filtered_bin_df = building_polygon_stats_filtered_df.copy()

# create bin edges and labels for accuracy category
bin_edges = [-np.inf, 59, 79, 89, 91, np.inf]
bin_labels = ["<59", "60-79", "80-89", "90-91", ">91"]

# create accuracy category column for tor tents
tent_filtered_bin_df["accuracy_category"] = pd.cut(
    tent_filtered_bin_df["accuracy_percentage_tent"],
    bins=bin_edges,
    labels=bin_labels,
    right=False,
)

tent_filtered_bin_df

# %%
# group df by filename and area for accuracy category
category_counts = (
    tent_filtered_bin_df.groupby(["filename", "area"])["accuracy_category"]
    .value_counts()
    .unstack(fill_value=0)
)

tent_category_counts = category_counts.sort_values(by=">91", ascending=False)
tent_category_counts

# %% [markdown]
# #### Save tent percentages

# %%
# transform index to columns so they appear in csv
tent_category_counts.reset_index(inplace=True)

tent_category_csv_filename = runid + "_tent_percentage_accuracy.csv"
tent_category_csv_file_path = outputs_dir / tent_category_csv_filename

# save as csv into outputs_dir
tent_category_counts.to_csv(tent_category_csv_file_path, index=False)

# %% [markdown]
# ### Tent Visualisations

# %%
category_counts = category_counts.sort_values(by="<59", ascending=False)
category_counts_reset = category_counts.reset_index()

unique_areas = category_counts_reset["area"].unique()

plt.rc("font", size=8)


for area in unique_areas:

    area_data = category_counts_reset[category_counts_reset["area"] == area]

    plt.figure(figsize=(12, 8))
    area_data.set_index("filename").drop("area", axis=1).plot(
        kind="barh", stacked=True, colormap="viridis"
    )

    plt.xlabel("count")
    plt.ylabel("")
    plt.title(f"accuracy categories for {area}")

    plt.legend(title="accuracy bin", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()


# %%
filtered_category_counts = category_counts[[">91", "<59"]]
filtered_category_counts = filtered_category_counts.sort_values(
    by="<59", ascending=True
)
filtered_category_counts.reset_index(inplace=True)

plt.rc("font", size=8)

plt.figure(figsize=(12, 16))

sns.set(style="whitegrid")

for i in range(len(filtered_category_counts)):
    plt.plot(
        [
            filtered_category_counts[">91"].iloc[i],
            filtered_category_counts["<59"].iloc[i],
        ],
        [filtered_category_counts["filename"].iloc[i]] * 2,
        color="gray",
        linestyle="--",
    )

plt.scatter(
    filtered_category_counts[">91"],
    filtered_category_counts["filename"],
    color="green",
    label=">91",
    s=100,
)
plt.scatter(
    filtered_category_counts["<59"],
    filtered_category_counts["filename"],
    color="red",
    label="<59",
    s=100,
)

plt.xlabel("count")
plt.title("Accuracy count for each training tile")

plt.legend()

plt.show()


# %% [markdown]
# ### Buildings accuracy percentage

# %%
# group df by csv name, aggregate stats for buildings and reset index
building_stats_grouped_df = (
    merged_clean_building_polygon_stats_df.groupby("csv_name")[
        "accuracy_percentage_building"
    ]
    .agg(["min", "max", "mean"])
    .reset_index()
)

# rename columns
building_stats_grouped_df.columns = [
    "csv_name",
    "min_accuracy_building",
    "max_accuracy_building",
    "avg_accuracy_building",
]

building_stats_grouped_df


# %%
# make copy of filtered_df
building_filtered_bin_df = building_polygon_stats_filtered_df.copy()

# create bin edges and labels for accuracy category
bin_edges = [-np.inf, 59, 79, 89, 91, np.inf]
bin_labels = ["<59", "60-79", "80-89", "90-91", ">91"]

building_filtered_bin_df["accuracy_category"] = pd.cut(
    building_filtered_bin_df["accuracy_percentage_building"],
    bins=bin_edges,
    labels=bin_labels,
    right=False,
)

building_filtered_bin_df

# %%
# group df by filename and area for accuracy category
building_category_counts = (
    building_filtered_bin_df.groupby(["filename", "area"])["accuracy_category"]
    .value_counts()
    .unstack(fill_value=0)
)

building_category_counts = building_category_counts.sort_values(
    by=">91", ascending=False
)
building_category_counts

# %% [markdown]
# #### Save building percentages

# %%
building_category_csv_filename = runid + "_building_percentage_accuracy.csv"

building_category_csv_file_path = outputs_dir / building_category_csv_filename

# transform index to columns so they appear in csv
building_category_counts.reset_index(inplace=True)

# save as csv
building_category_counts.to_csv(building_category_csv_file_path, index=False)

# %%
