import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def upd_single_track_model(delta_rad, v_mps, tS_s, lf_m, lr_m, c_alpha, mass, inertia, p1_sol_old, p2_sol_old,
                           psi_sol_old, dot_psi_sol_old, beta_sol_old):
    """
    Simulation of dynamics with linear single track model.
    Equations taken from https://de.wikipedia.org/wiki/Einspurmodell.
    """

    p1_sol = p1_sol_old + tS_s * v_mps * (math.cos(psi_sol_old - beta_sol_old))
    p2_sol = p2_sol_old + tS_s * v_mps * (math.sin(psi_sol_old - beta_sol_old))
    psi_sol = psi_sol_old + tS_s * dot_psi_sol_old
    a11 = -2 * c_alpha / (mass * v_mps)
    a12 = 1 - c_alpha * (lr_m - lf_m) / (mass * v_mps ** 2)
    b1 = -c_alpha / (mass * v_mps)
    a21 = -c_alpha * (lr_m - lf_m) / inertia
    a22 = -c_alpha * (lr_m ** 2 + lf_m ** 2) / (inertia * v_mps)
    b2 = c_alpha * lf_m / inertia
    beta_sol = beta_sol_old + tS_s * (a11 * beta_sol_old + a12 * dot_psi_sol_old + b1 * delta_rad)
    dot_psi_sol = dot_psi_sol_old + tS_s * (a21 * beta_sol_old + a22 * dot_psi_sol_old + b2 * delta_rad)

    return p1_sol, p2_sol, psi_sol, dot_psi_sol, beta_sol
