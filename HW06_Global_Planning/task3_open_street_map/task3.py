"""
Goal of Task 3:
    Understand how to use Open Street Map Data.

Hint: Look into https://osmnx.readthedocs.io/en/stable/index.html to check the osmnx package.
"""


import osmnx as ox
import os
import sys
import networkx as nx
import plotly.graph_objects as go
from osmnx import distance

import numpy as np

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)


def plot_path(lat, long, origin_point, destination_point):
    fig = go.Figure(go.Scattermapbox(
        name="Path",
        mode="lines",
        lon=long,
        lat=lat,
        marker={'size': 10},
        line=dict(width=4.5, color='blue')))  # adding source marker
    fig.add_trace(go.Scattermapbox(
        name="Source",
        mode="markers",
        lon=[origin_point[1]],
        lat=[origin_point[0]],
        marker={'size': 12, 'color': "red"}))

    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name="Destination",
        mode="markers",
        lon=[destination_point[1]],
        lat=[destination_point[0]],
        marker={'size': 12, 'color': 'green'}))

    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)  # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
                      mapbox_center_lat=30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox={
                          'center': {'lat': lat_center,
                                     'lon': long_center},
                          'zoom': 13})
    fig.show()


def main(place_name, start_name, end_name):
    """
    Calculate the shortest path from start to end location.

    inputs:
        place_name (type: str): center of the used map
        start_name: (type: tuple): coordinates of start location
        end_name: (type: tuple): coordinates of end location

    outputs:
        travel_time (type: float): travel time of the shortest path [in s]
        route_distance (type: float): distance of the shortest path [in m]
    """

    ox.config(use_cache=True, log_console=False)
    # The next line imports the graph of the location with a radius of 1500 meters.
    graph = ox.graph_from_address(place_name, dist=1500, network_type='drive')
    # How this graph looks like can be seen in the following figure.
    fig, ax = ox.plot_graph(graph)
    # The next two lines define the start and destination locations.

    # Find close nodes of the locations in the graph.
    start = distance.get_nearest_node(graph, start_name)
    end = distance.get_nearest_node(graph, end_name)

    # Extract the shortest path and plot the route.
    route = nx.shortest_path(graph, start, end, weight='travel_time')
    long = []
    lat = []
    for i in route:
        point = graph.nodes[i]
        long.append(point['x'])
        lat.append(point['y'])
    plot_path(lat, long, (graph.nodes[start]['x'], graph.nodes[start]['y']),
              (graph.nodes[end]['x'], graph.nodes[end]['y']))

    # Task:
    # ToDo: Calculate the travel time (in seconds) and the distance (in meters) to get from the Stiglmaierplatz
    #   to the ProfHuberPlatz.
    # Hint: This is the free-flow travel time. No traffic is considered.
    ########################
    #  Start of your code  #
    ########################

    travel_time = 0.0
    route_distance = 0.0

    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    origin_node = ox.get_nearest_node(graph, start_name)
    destination_node = ox.get_nearest_node(graph, end_name)
    path_distance = nx.shortest_path(graph, origin_node, destination_node, weight='length')
    route_distance = float(sum(ox.utils_graph.get_route_edge_attributes(graph, path_distance, 'length')))

    path_time = nx.shortest_path(graph, origin_node, destination_node, weight='travel_time')
    travel_time = float(sum(ox.utils_graph.get_route_edge_attributes(graph, path_time, 'travel_time')))

    ########################
    #   End of your code   #
    ########################

    print("The rounded travel time is: " + str(round(travel_time)) + " Seconds")
    print("The rounded route distance is: " + str(round(route_distance)) + " Meters")
    return travel_time, route_distance


if __name__ == "__main__":
    # The center of the map we are using in the following task is the center of maxvorstadt, munich.
    stiglmaierplatz = (48.14765, 11.55877)
    profhuberplatz = (48.150120, 11.581145)
    place_name_ = "maxvorstadt"
    travel_time, route_distance = main(place_name_, stiglmaierplatz, profhuberplatz)
