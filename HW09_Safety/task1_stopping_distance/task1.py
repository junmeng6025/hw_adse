"""
Goal of Task 1:
    Implement a function that calculates the Difference of Space distance and Stopping distance (DSS).

Hint:
    In order to properly run the code in this homework, additional packages have to be installed via pip.
    Use:
        `pip install trajectory-planning-helpers`
        `pip install scenario-testing-tools`
"""


import numpy as np


def calc_dss(pos_ego, vel_ego, pos_obj, vel_obj, veh_len=4.7, a_max=15.0):
    """
    inputs:
        pos_ego (type: np.ndarray): position of ego vehicle as numpy array with columns x, y [in m]
        vel_ego (type: float): velocity of ego vehicle [in m/s]
        pos_obj (type: np.ndarray): position of object vehicle as numpy array with columns x, y [in m]
        vel_obj (type: float): velocity of object vehicle [in m/s]
        veh_len (type: float): (optional) vehicle length (assumed identical for both) [in m]
        a_max (type: float): (optional) maximum acceleration [in m/s^2]

    output:
        dss (type: np.float64): difference of space distance and stopping distance [in m]
    """

    # Task:
    # ToDo: Calculate the Difference of Space distance and Stopping distance (DSS) for a given ego
    #   vehicle (pos, vel) and an object vehicle (pos, vel).
    # Hints:
    #   - The formula is given in the slides of the lecture 09_Safety.
    #   - The maximum acceleration is assumed to be 15.0 m/s^2.
    #   - Assumption: the provided object vehicle is _in front_ of the ego vehicle.
    ########################
    #  Start of your code  #
    ########################

    dx = np.linalg.norm(pos_ego - pos_obj)
    dss = vel_obj ** 2 / (2 * a_max) + dx - vel_ego ** 2 / (2 * a_max) - veh_len

    ########################
    #   End of your code   #
    ########################

    return dss


if __name__ == "__main__":
    dss_test = calc_dss(pos_ego=np.array([0.0, 0.0]), vel_ego=50.0, pos_obj=np.array([44.7, 0.0]), vel_obj=40.0)
    print("The calculated DSS is:")
    print(dss_test)
    test = np.isclose(dss_test, 10.0)
    assert test, "Calculated DSS is wrong."
