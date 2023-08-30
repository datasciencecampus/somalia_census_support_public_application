# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# import standard and third party libraries
from pathlib import Path
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from IPython.display import IFrame

# %%
data_dir = Path.cwd().parent.joinpath("data")
file_path = data_dir.joinpath("somalia_idp_sites_march23.csv")
shelter_file_path = data_dir.joinpath("SOM_IDP_Site_Monitoring_July2023.csv")

# %% [markdown]
# ## All camps

# %%
# camp location data
data = pd.read_csv(file_path)
data = data.dropna(subset=["Latitude", "Longitude"])
data.columns = data.columns.str.lower().str.replace(" ", "_")
data.head()

# %% [markdown]
# ### All camps map

# %%
somalia_map = folium.Map(location=[5.152, 45.338], zoom_start=6)

marker_cluster = MarkerCluster().add_to(somalia_map)

original_locations = [
    {"name": "Baidoa", "lat": 3.11442, "lon": 43.65199},
    {"name": "Beledweyne", "lat": 4.74441, "lon": 45.19848},
    {"name": "Mogadishu", "lat": 2.07327, "lon": 45.33481},
    {"name": "Kismayo", "lat": -0.358, "lon": 42.55459},
    {"name": "Doolow", "lat": 4.16424, "lon": 42.07847},
]

original_locations2 = [
    {"name": "Dhuusamarreeb", "lat": 5.53089, "lon": 46.39878},
    {"name": "Hargeisa", "lat": 9.56868, "lon": 44.07593},
    {"name": "Bossaso", "lat": 11.27703, "lon": 49.18856},
    {"name": "Galkacyo", "lat": 6.79063, "lon": 47.43720},
    {"name": "Burao", "lat": 9.53164, "lon": 45.54605},
]

for index, row in data.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]], popup=row["idp_site"]
    ).add_to(marker_cluster)


for location in original_locations:
    folium.Marker(
        location=[location["lat"], location["lon"]],
        popup=location["name"],
        icon=folium.Icon(color="darkblue", icon="location-pin"),
    ).add_to(somalia_map)

for location in original_locations2:
    folium.Marker(
        location=[location["lat"], location["lon"]],
        popup=location["name"],
        #     icon=folium.DivIcon(
        #         icon_size=(150,36),
        #         icon_anchor=(75,18),
        #         html='<div style="background-color:lightblue; boreder:2px solid blue; border-radius:4px; padding:6px;">b>' + location['name'] + '</b></div>'
        #     )
        icon=folium.Icon(color="darkpurple", icon="location-pin"),
    ).add_to(somalia_map)
somalia_map

# %%
# saves map as html - saved in src folder because I haven't updated path

map_file = "somalia_map.html"
somalia_map.save(map_file)

IFrame(src=map_file, width=800, height=600)

# %% [markdown]
# ## Shelter type

# %%
# shelter location data
shelter_data = pd.read_csv(shelter_file_path)
shelter_data.columns = shelter_data.columns.str.lower().str.replace(" ", "_")

shelter_data.head()

# %%
# renaming column names
shelter_data.rename(
    columns={
        "how_many_households_are_present_at_the_site?": "households",
        "how_many_individuals_are_present_at_the_site?": "individuals",
        "what_type_of_shelter_is_most_common_within_the_idp_site?": "shelter_type",
        "settlement/cluster/umbrella/village/section": "settlement",
    },
    inplace=True,
)

shelter_mapping = {
    "Emergency Shelters (Somali Traditional House/ Buul/ Tent/ Emergency Shelter Kits/ Timber and Plastic Sheet with CGI Roof)": "emergency",
    "Transitional Shelters (Mundul/Baraako/Plywood wall with CGI roofing, CGI sheet wall and roof)": "transitional",
    "Permanent Shelters (Stone brick wall with CGI roofing, Mud block shelter)": "permanent",
}

shelter_data["shelter_type"] = shelter_data["shelter_type"].replace(shelter_mapping)

shelter_data.head()

# %% [markdown]
# #### Shelter map

# %%
shelter_colors = {"emergency": "orange", "transitional": "blue", "permanent": "green"}

# %%
# Create a base map centered on Somalia
shelter_map = folium.Map(location=[5.152149, 46.199616], zoom_start=6)

# Add data points to the map
for index, row in shelter_data.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        # radius=row["households"] / 300,
        color=None,  # Let the color be determined by the 'shelter_type' column
        fill=True,
        fill_color=shelter_colors.get(row["shelter_type"], "gray"),
        fill_opacity=0.7,
        popup=row["idp_site"],
    ).add_to(shelter_map)

shelter_map


# %%
# Group by district and shelter type, and calculate sums
grouped_data = (
    shelter_data.groupby(["district", "shelter_type"])
    .agg({"households": "sum", "individuals": "sum"})
    .reset_index()
)
grouped_data

# %%
# Create pivot tables for households and individuals
pivot_households = grouped_data.pivot(
    index="district", columns="shelter_type", values="households"
).fillna(0)
pivot_individuals = grouped_data.pivot(
    index="district", columns="shelter_type", values="individuals"
).fillna(0)

# Create bar plots for households and individuals
pivot_households.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=[shelter_colors[col] for col in pivot_households.columns],
)
plt.title("Households by Shelter Type")
plt.ylabel("Number of Households")
plt.xlabel("District")
plt.legend(title="Shelter Type")
plt.show()

pivot_individuals.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=[shelter_colors[col] for col in pivot_households.columns],
)
plt.title("Individuals by Shelter Type")
plt.ylabel("Number of Individuals")
plt.xlabel("District")
plt.legend(title="Shelter Type")
plt.show()

# %%
# Group by district and shelter type, and calculate sums
all_grouped_data = (
    shelter_data.groupby(["shelter_type"])
    .agg({"households": "sum", "individuals": "sum"})
    .reset_index()
)
all_grouped_data

# %%
# Set custom colors for pie chart
colors = ["orange", "blue", "green"]

# Create pie chart for households
plt.figure(figsize=(8, 5))
plt.pie(
    all_grouped_data["households"],
    labels=all_grouped_data["shelter_type"],
    colors=colors,
    autopct="%1.1f%%",
)
plt.title("Distribution of Shelter Types for Households")
plt.show()

# Create pie chart for individuals
plt.figure(figsize=(8, 5))
plt.pie(
    all_grouped_data["individuals"],
    labels=all_grouped_data["shelter_type"],
    colors=colors,
    autopct="%1.1f%%",
)
plt.title("Distribution of Shelter Types for Individuals")
plt.show()

# %% [markdown]
# ## Camp extent modelling
#

# %%
data.head()

# %%
data.rename(
    columns={
        "_hh_(q1-2023)_": "households",
        "_individual_(q1-2023)_": "individuals",
        "date_idp_site_established": "established",
    },
    inplace=True,
)

# %%
extents_data = data[["region", "district", "latitude", "longitude", "households"]]

extents_data.head()

# %%
extents_data["households"] = pd.to_numeric(extents_data["households"], errors="coerce")

# %%
# calculating area

building_area = 14  # in square meters
space_between_buildings = 2  # in meters

extents_data["total_building_area"] = (
    extents_data["households"] * (building_area + space_between_buildings)
    - space_between_buildings
)

extents_data

# %%
from shapely.geometry import Point

# %%
extents_data["geometry"] = None


# %%
for index, row in extents_data.iterrows():
    lon = row["longitude"]
    lat = row["latitude"]
    total_area = pd.to_numeric(row["total_building_area"])

    # Create a rectangular polygon based on half the width and half the height
    half_width = (total_area / (building_area + space_between_buildings)) / 2
    half_height = (building_area + space_between_buildings) / 2

    # Define the coordinates of the polygon's corners
    coords = [
        (lon - half_width, lat - half_height),
        (lon + half_width, lat - half_height),
        (lon + half_width, lat + half_height),
        (lon - half_width, lat + half_height),
    ]

    # Create a Polygon object using the coordinates
    polygon = Point(coords).envelope

    # Assign the polygon to the 'geometry' column
    extents_data.at[index, "geometry"] = polygon


# %%
from shapely.geometry import Polygon

for index, row in extents_data.iterrows():
    lon = row["longitude"]
    lat = row["latitude"]
    total_area = row["total_building_area"]

    half_width = total_area / (building_area + space_between_buildings) / 2
    half_height = (building_area + space_between_buildings) / 2

    coords = [
        (lon - half_width, lat - half_height),
        (lon + half_width, lat - half_height),
        (lon + half_width, lat + half_height),
        (lon - half_width, lat + half_height),
    ]

    polygon = Polygon(coords)  # Create a Polygon object using the coordinates

    extents_data.at[index, "geometry"] = polygon


# %%
