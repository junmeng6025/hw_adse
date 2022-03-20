"""
Goal of Task 2:
    Implement a function that identifies if a provided path collides with a track boundary.

Hint:
    In order to properly run the code in this homework, additional packages have to be installed via pip.
    Use:
        `pip install trajectory-planning-helpers`
        `pip install scenario-testing-tools`
"""


import numpy as np
import shapely.geometry


def check_bound_collision(path, bound, veh_width=2.8):
    """
    inputs:
        path (type: np.ndarray): path of the ego trajectory with columns x, y [in m]
        bound (type: np.ndarray): track boundary with columns x, y [in m]
        veh_width (type: float): (optional) vehicle width [in m]

    output:
    intersection (type: bool): flag indicating intersection of the two provided paths (True if intersection exists)
    """

    # Task:
    # ToDo: Identify if a provided path collides with a track boundary. Both, the path and the boundary, are provided
    #  as a sequence of coordinates connected by straights. This function can then be used to detect collisions with
    #  any of the track boundaries. Cases where the trajectory itself does not hit a bound, but a side of the vehicle
    #  would touch the boundary must also be detected.
    # Hints:
    #   - For this implementation we recommend the library "shapely" (documentation: shapely.readthedocs.io) with the
    #     following useful methods for this task:
    #       - buffer (inflate a given shapely object for a specified increment)
    #       - intersect (check whether two shapely objects intersect at any point)
    #   - the provided array (bound and path) can be directly converted to a shapely object via
    #     'shapely.geometry.LineString(path)'.
    #   - It should be noted, that the problem is significantly reduced in complexity for this course, the following
    #     assumptions hold:
    #       - The vehicle length is not considered
    #       - The vehicle kinematics are not considered (e.g. sheer off behavior)
    ########################
    #  Start of your code  #
    ########################

    path_bound = shapely.geometry.LineString(path).buffer(veh_width / 2)
    bound_obj = shapely.geometry.LineString(bound)
    intersection = path_bound.intersects(bound_obj)

    ########################
    #   End of your code   #
    ########################

    return intersection


if __name__ == "__main__":
    path = np.array([[10.0, 10.0],
                     [10.0, 20.0],
                     [15.0, 30.0],
                     [25.0, 35.0],
                     [35.0, 35.0]])
    bound = np.array([[20.0, 10.0],
                      [20.0, 20.0],
                      [25.0, 30.0],
                      [35.0, 40.0]])
    assert check_bound_collision(path=path,
                                 bound=bound)
