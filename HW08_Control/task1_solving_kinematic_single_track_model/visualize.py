import numpy as np
import matplotlib.pyplot as plt


def visualize_vehicle_path(p1, p2, psi):
    """
    Function that visualizes the solution of the "kinematic single track model" task.

    inputs:
        p1 (type: np.ndarray, shape: (N,)): global vehicle positions p1
        p2 (type: np.ndarray, shape: (N,)): global vehicle positions p2
        psi (type: np.ndarray, shape: (N,)): global headings
    """

    N = p1.shape[0]
    rel_arrow_width = (abs(max(p1) - min(p1)) + abs(max(p2) - min(p2))) / 2 * 0.025

    # Plot
    plt.plot(p1, p2, 'b-')
    # plt.rcParams["font.family"] = "sans-serif"
    # plt.rcParams["font.sans-serif"] = "Verdana"
    plt.grid()
    plt.xlabel('p_1')
    plt.ylabel('p_2')
    plt.title("Driven Path")
    plt.arrow(p1[int(N / 2)], p2[int(N / 2)], np.cos(psi[int(N / 2)]), np.sin(psi[int(N / 2)]), lw=0,
              head_starts_at_zero=True, length_includes_head=True, head_length=2 * rel_arrow_width,
              head_width=rel_arrow_width, color='b')
    plt.show()
