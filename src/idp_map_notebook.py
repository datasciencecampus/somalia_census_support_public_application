# ---
# jupyter:
#   jupytext:
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

# %%
# import standard and third party libraries
from pathlib import Path
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from IPython.display import IFrame

# %%
data_dir = Path.cwd().parent.joinpath("data")

file_path = data_dir.joinpath("somalia_idp_sites_march23.csv")

# %%
data = pd.read_csv(file_path)
data

# %%
data = data.dropna(subset=["Latitude", "Longitude"])
data

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
        location=[row["Latitude"], row["Longitude"]], popup=row["Neighbourhood"]
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
map_file = "somalia_map.html"
somalia_map.save(map_file)

IFrame(src=map_file, width=800, height=600)

# %%
