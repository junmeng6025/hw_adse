"""
Goal of Task 2:
    Implement a lateral driving controller.
"""


import math
import numpy as np
import matplotlib.pyplot as plt
from simulate import upd_single_track_model


def control(d_m, d_m_old, v_mps, radius_m, tS_s, lf_m, lr_m):
    """
    Function that controls the lateral driving behavior by steering.

    inputs:
        d_m (type: np.float64): distance from center of gravity of the vehicle to trajectory from current timestep
        d_m_old (type: np.float64): distance from center of gravity of the vehicle to trajectory from previous timestep
        v_mps (type: np.float64): vehicle speed from current timestep
        radius_m (type: np.float64): curve radius of the vehicle CoG
        tS_s (type: np.float64): stepsize in seconds
        lf_m (type: np.float64): distance from CoG to front axle
        lr_m (type: np.float64): distance from CoG to rear axle

    output:
        steering (type: np.ndarray, shape: (N,)): steering angle in rad based on single track model
    """

    # Task:
    # ToDo: Implement a lateral driving controller to track a circular path with given radius.
    # Hints:
    #   - use a mix between feedforward and feedback control
    #   - use numerical derivation to obtain the derivative of d_m
    #   - slide page 39
    ########################
    #  Start of your code  #
    ########################

    k1 = 15
    k2 = 7
    dd_m = (d_m - d_m_old) / tS_s
    kp = 1 / radius_m
    kc = kp - (k1 * d_m + k2 * dd_m) / (v_mps ** 2)
    steering = kc * (lf_m + lr_m)

    ########################
    #   End of your code   #
    ########################

    return steering


def solve_closed_loop(radius_m, v_mps):
    # number of steps to simulate
    N = 500
    # step size in seconds
    tS_s = 0.02
    # distance from CoG to front axle
    lf_m = 2
    # distance from CoG to rear axle
    lr_m = 2.2
    # vehicle mass
    mass = 1000
    # vehicle inertia
    inertia = 1200
    # cornering stiffness
    c_alpha = 80000

    # initialize solution vectors
    p1_sol = np.zeros(N)
    p2_sol = np.zeros(N)
    psi_sol = np.zeros(N)
    dot_psi_sol = np.zeros(N)
    beta_sol = np.zeros(N)
    d_sol = np.zeros(N)

    # specify initial conditions such that the
    # vehicle starts tangential to the target circle
    p1_sol[0] = radius_m
    p2_sol[0] = 0
    psi_sol[0] = math.pi / 2
    dot_psi_sol[0] = 1 / radius_m * v_mps
    beta_sol[0] = 0

    # solve differential equation for inputs delta_rad and v_mps
    # use euler forward integration similar to the exercise
    # see lecture slides for kinematic single track model
    d_m = 0
    # track d_max_m
    d_max_m = 0
    for i in range(1, p1_sol.shape[0]):
        # store old tracking error for control law
        d_m_old = d_m
        # determine lateral tracking error
        # caution! this only works due to the tracking of a circle!
        d_m = radius_m - np.sqrt(p1_sol[i - 1]**2 + p2_sol[i - 1]**2)
        # store error
        d_sol[i] = d_m
        # update maximum control error
        d_max_m = max(abs(d_m), d_max_m)
        # calculate steering angle
        delta_rad = control(d_m, d_m_old, v_mps, radius_m, tS_s, lf_m, lr_m)
        p1_sol[i], p2_sol[i], psi_sol[i], dot_psi_sol[i], beta_sol[i] = \
            upd_single_track_model(delta_rad, v_mps, tS_s,
                                   lf_m, lr_m, c_alpha, mass, inertia,
                                   p1_sol[i - 1], p2_sol[i - 1], psi_sol[i - 1],
                                   dot_psi_sol[i - 1], beta_sol[i - 1])

    print('The maximum control error was: {:4.2f}m'.format(d_max_m))
    return p1_sol, p2_sol, psi_sol, d_max_m, d_sol


if __name__ == "__main__":
    # straight driving
    p1, p2, psi, _, d_m = solve_closed_loop(20, 10)
    plt.plot(p1, p2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.xlabel('East coordinate in m')
    plt.ylabel('North coordinate in m')
    plt.show()
    plt.plot(d_m)
    plt.grid()
    plt.xlabel('Steps')
    plt.ylabel('Lateral error in m')
    plt.show()
