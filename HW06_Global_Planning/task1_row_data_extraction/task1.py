"""
Goal of Task 1:
    Get an overview of how a simple graph class can be implemented. This is a precondition to perform the dijkstra's
    or A* Algorithm. We need to extract the raw data of the map and use it to perform the optimization.

Hint:
    This is a very simple example of how the edges can be represented. In industrial applications more complicated
    raw data is used.
"""


class Graph:
    def __init__(self, map_raw_data):
        """
        This function reads out our map information. The map is defined via edges and their costs.
        The first location in the .txt-file is the start location of an edge and the second the goal location.
        The number after these two locations on the edge is the cost value to get from start to goal.
        """

        graph_edges = []

        # Subtask 1:
        # ToDo: extract the information of the .txt-file
        # Hints:
        #   - The result should be a list of tuples:  graph_edges =[('LocationA_from', 'Location_to', distance), ...]
        #   - The output are the edges from A to B and the corresponding distance for this edge.
        ########################
        #  Start of your code  #
        ########################

        f = open(map_raw_data)
        for line in f:
            data = line.strip('\n').split(' ')
            graph_edges.append(data)

        ########################
        #   End of your code   #
        ########################

        self.nodes = set()

        # Subtask 2:
        # ToDo: add two more attributes (self.nodes and self.neighbors_list) to the class
        # Hints:
        #   - self.nodes --> set of strings {'Location1', 'Location2', ...}
        #   - self.neighbors_list saves all adjacent nodes for every node and the distance to them. It should be
        #   realised as a dict of sets. The sets contain tuples of the form --> (str('Location'), float(distance)).
        #   Example:
        #   self.neighbors_list --> dict: {'Location1': {('Location_x', distance_to_x),('Location_y', distance_to_y)...
        #                                  'Location2': {('Location_z', distance_to_z),('Location_p', distance_to_p)...
        #                                   ...}
        ########################
        #  Start of your code  #
        ########################

        for location in graph_edges:
            self.nodes.add(location[0])
            self.nodes.add(location[1])
        self.neighbors_list = {}
        self.neighbors_list = dict.fromkeys(self.nodes)
        for edge in graph_edges:
            self.neighbors_list[edge[0]] = set()
        for edge in graph_edges:
            self.neighbors_list[edge[0]].add((edge[1], float(edge[2])))

        ########################
        #   End of your code   #
        ########################


def generate_graph(file):
    graphs = Graph(file)
    return graphs


if __name__ == "__main__":
    graph = generate_graph("munich.txt")
    print(graph.nodes)
    print(graph.neighbors_list)
    """ The print functions should print the existing locations/nodes (graph.nodes)
     and the nodes (graph.neighbors_list) with all neighbors and the distance to them """
