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

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from functions_library import get_folder_paths

# %%
folder_dict = get_folder_paths()
# set model and output directories
outputs_dir = Path(folder_dict["outputs_dir"])

# %%
# create df of csv building counts
file_pattern = "*_building_polygon_stats.csv"
csv_files = outputs_dir.glob(file_pattern)

dataframes = []

for csv_file in csv_files:
    df = pd.read_csv(
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
    df["area"] = df["filename"].str.split("_").str[2]
    # adding csv file name as column
    df["csv_name"] = csv_file.stem
    dataframes.append(df)


merged_df = pd.concat(dataframes, ignore_index=True)
# removing background tiles
merged_df = merged_df[~merged_df["filename"].str.endswith("_background")]

merged_df

# %%
# find rows where 'accuracy_percentage_tent' is -inf
inf_rows = merged_df[merged_df["accuracy_percentage_tent"] == -np.inf]
# remove rows where 'accuracy_percentage_tent' is -inf
merged_clean_df = merged_df[merged_df["accuracy_percentage_tent"] != -np.inf]

merged_clean_df

# %%
grouped_df = (
    merged_clean_df.groupby("csv_name")["accuracy_percentage_tent"]
    .agg(["min", "max", "mean"])
    .reset_index()
)

grouped_df.columns = [
    "csv_name",
    "min_accuracy_tent",
    "max_accuracy_tent",
    "avg_accuracy_tent",
]

grouped_df


# %%
exclude_filenames = [
    "border_testing_2023-12-20_1108_building_polygon_stats",
    "border_testing_2024-01-15_1733_building_polygon_stats",
    "border_testing_2024-01-15_1843_building_polygon_stats",
]

# removing low building count csvs
filtered_df = merged_clean_df[~merged_clean_df["csv_name"].isin(exclude_filenames)]

filtered_df

# %%
filtered_bin_df = filtered_df.copy()

bin_edges = [-np.inf, 59, 79, 89, 91, np.inf]
bin_labels = ["<59", "60-79", "80-89", "90-91", ">91"]

filtered_bin_df["accuracy_category"] = pd.cut(
    filtered_bin_df["accuracy_percentage_tent"],
    bins=bin_edges,
    labels=bin_labels,
    right=False,
)
filtered_bin_df

# %% jupyter={"outputs_hidden": true}
category_counts = (
    filtered_bin_df.groupby(["filename", "area"])["accuracy_category"]
    .value_counts()
    .unstack(fill_value=0)
)
category_counts = category_counts.sort_values(by=">91", ascending=False)
category_counts

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


# %%
