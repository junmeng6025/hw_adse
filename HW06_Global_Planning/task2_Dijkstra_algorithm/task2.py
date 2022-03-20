"""
Goal of Task 2:
    Implementation of a route search algorithm.

Hint: refer to lecture
"""


from collections import deque
import sys
import os

inf = float("inf")
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)


def dijkstra(nodes, neighbors_list, start_node, end_node):
    """
    This function is the core of the Dijkstra's Algorithm.

    inputs:
        nodes (type: set): nodes of the given route network, cf. to task1.py
        neighbors_list (type: dict): adjacent nodes for every node and the distance to them, cf. to task1.py
        start_node (type: str): "where we are"
        end_node (type: str): "where we want to go"

    outputs:
        path (type: deque): contains all nodes of the optimal path in the correct order
        opt_distance (type: float): distance of optimal path
    """

    global temp
    unvisited_nodes = nodes.copy()  # All nodes are unvisited.

    # Subtask 1:
    # ToDo: set distance of start node to 0.0 and distance of all other nodes to infinity
    # Hint: distance values should be floats
    ########################
    #  Start of your code  #
    ########################

    distance_from_start = dict()
    for node in nodes:
        if node == start_node:
            distance_from_start[node] = 0
        else:
            distance_from_start[node] = inf

    ########################
    #   End of your code   #
    ########################

    # Initialize previous_node, the dictionary that maps each node to the
    # node it was visited from when the the shortest path to it was found.
    previous_node = {node: None for node in nodes}
    optimal_distance = 0

    while unvisited_nodes:
        current_node = min(
            unvisited_nodes, key=lambda node: distance_from_start[node]
        )
        unvisited_nodes.remove(current_node)

        if distance_from_start[current_node] == inf:
            break

        # Subtask 2:
        # ToDo: Calculate the 'new_path' for every neighbor in 'neighbors_list[current_node]'. If the 'new_path' is
        #   shorter than the current distance from start to this neighbor node, you have to update the previous node
        #   and the distance from start.
        # Hints:
        #   - use the dict 'distance_from_start' which includes the location and the corresponding distance
        #   - use the dict 'previous_node' with location1 and corresponding location2 as string
        ########################
        #  Start of your code  #
        ########################

        distance = inf
        for neighbor in neighbors_list[current_node]:
            if neighbor[0] == end_node:
                distance = neighbor[1]
                temp = neighbor
            elif neighbor[0] in unvisited_nodes:
                if distance > neighbor[1]:
                    distance = neighbor[1]
                    temp = neighbor
        optimal_distance += temp[1]
        distance_from_start[temp[0]] = optimal_distance
        previous_node[temp[0]] = current_node

        ########################
        #   End of your code   #
        ########################

        if current_node == end_node:
            break

    path = deque()
    current_node = end_node
    print(previous_node)

    # Subtask 3:
    # ToDo: Extract the optimal path and the optimal distance.
    # Hints:
    #   - use deque: https://docs.python.org/3/library/collections.html#collections.deque
    #   - path should look like: deque(['Location1','Location2',...])
    ########################
    #  Start of your code  #
    ########################

    while current_node != start_node:
        if previous_node[current_node]:
            path.appendleft(previous_node[current_node])
            current_node = previous_node[current_node]
            print(current_node)
    path.append(end_node)

    ########################
    #   End of your code   #
    ########################

    opt_distance = distance_from_start[end_node]

    return path, opt_distance


if __name__ == "__main__":
    # load dummy input
    neighbors_list_ = dict({'HBF': {('Odeonsplatz', 14.0), ('Königsplatz', 6.0), ('Stiglmaierplatz', 8.0)},
                            'Eisbach': {('ProfHuberPlatz', 10.0), ('Odeonsplatz', 9.0)},
                            'Karolinenplatz': {('Königsplatz', 4.0), ('Odeonsplatz', 6.0)},
                            'Pinakothek': {('ProfHuberPlatz', 9.0), ('Königsplatz', 7.0), ('Theresienstraße', 5.0)},
                            'Königsplatz': {('Stiglmaierplatz', 5.0), ('Karolinenplatz', 4.0),
                                            ('HBF', 6.0), ('Pinakothek', 7.0)},
                            'Stiglmaierplatz': {('Theresienstraße', 7.0), ('Königsplatz', 5.0),
                                                ('HBF', 8.0)},
                            'ProfHuberPlatz': {('Odeonsplatz', 9.0), ('Eisbach', 10.0), ('Pinakothek', 9.0)},
                            'Theresienstraße': {('Pinakothek', 5.0), ('Stiglmaierplatz', 7.0)},
                            'Odeonsplatz': {('ProfHuberPlatz', 9.0), ('Karolinenplatz', 6.0),
                                            ('Eisbach', 9.0), ('HBF', 14.0)}})
    nodes_ = {'HBF', 'Eisbach', 'Karolinenplatz', 'Pinakothek', 'Königsplatz', 'Stiglmaierplatz',
              'ProfHuberPlatz', 'Theresienstraße', 'Odeonsplatz'}

    """ The following function calculates the optimal path and the optimal distance to the destination. """

    optimal_path, optimal_distance = dijkstra(nodes_, neighbors_list_, "Stiglmaierplatz", "Eisbach")
    print(optimal_path)
    print(optimal_distance)
