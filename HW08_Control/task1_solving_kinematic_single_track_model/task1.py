"""
Goal of Task 1:
    Implement the solution to the differential equation of the kinematic single track model.
"""


import math
import numpy as np
from visualize import visualize_vehicle_path


def solve_kinematic_bicycle(x0, delta_rad, v_mps):
    """
    Function that solves the differential equations of the kinematic bicycle model.

    inputs:
        x0 (type: np.ndarray, shape: (3,)): initial vehicle state
        delta_rad (type: np.ndarray, shape: (N,)): steering angle changes
        v_mps (type: np.ndarray, shape: (N,)): vehicle speed

    outputs:
        p1 (type: np.ndarray, shape: (N,)): global vehicle positions p1
        p2 (type: np.ndarray, shape: (N,)): global vehicle positions p2
        psi (type: np.ndarray, shape: (N,)): global headings
    """

    # step size in seconds
    tS = 0.1
    # distance from CoG to front axle
    lf_m = 2
    # distance from CoG to rear axle
    lr_m = 2.2

    # initialize solution vectors
    p1 = np.zeros(len(delta_rad))
    p2 = np.zeros(len(delta_rad))
    psi = np.zeros(len(delta_rad))

    # Subtask 1:
    # ToDo: specify initial conditions from x0
    # Hint: x0 is given in form of a vector with [p1, p2, psi]
    ########################
    #  Start of your code  #
    ########################

    [p1[0], p2[0], psi[0]] = x0

    ########################
    #   End of your code   #
    ########################

    # Subtask 2:
    # ToDo: solve differential equation for inputs delta_rad and v_mps
    # Hints:
    #   - use euler forward integration similar to the exercise
    #   - see lecture slides for kinematic single track model
    #   - slide page 32
    for i in range(1, p1.shape[0]):
        ########################
        #  Start of your code  #
        ########################
        beta = math.atan(math.tan(delta_rad[i]) * lr_m / (lf_m + lr_m))
        p1[i] = p1[i - 1] + tS * (v_mps[i] * math.cos(psi[i - 1]) + beta)
        p2[i] = p2[i - 1] + tS * (v_mps[i] * math.sin(psi[i - 1]) + beta)
        psi[i] = psi[i - 1] + tS * (math.tan(delta_rad[i]) * v_mps[i] / (lf_m + lr_m))
        continue

        ########################
        #   End of your code   #
        ########################

    return p1, p2, psi


if __name__ == "__main__":
    # generate test cases
    N = 100
    # straight driving
    delta_rad_test1 = np.full(N, 0)
    v_mps_test1 = np.full(N, 10)
    x0 = np.array([0, 10, 1])
    p1, p2, psi = solve_kinematic_bicycle(x0, delta_rad_test1, v_mps_test1)
    print('The solution is: ')
    print('p1: ' + str(p1))
    print('p2: ' + str(p2))
    print('psi: ' + str(psi))

    # visualize your solution:
    visualize_vehicle_path(p1, p2, psi)
