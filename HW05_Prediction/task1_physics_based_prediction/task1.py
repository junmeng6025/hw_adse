"""
Goal of Task 1:
    Implement the Constant Turn Rate and Acceleration (CTRA)-Model for Physics-based Prediction.
"""


import numpy as np
import matplotlib.pyplot as plt
from bin.task1_prep import get_next_sample, prepare_prediction, object_prediction


class PhysicsPrediction:
    def __init__(self):
        self.state_variables = ["t", "x", "y", "yaw", "v", "yawRate", "a"]

    def step(self, input_tracking=None):
        """
        DO NOT MODIFY THIS FUNCTION
        """

        if input_tracking is None:
            input_tracking = get_next_sample()

        prediction_input_list = prepare_prediction(self.state_variables, input_tracking)
        physics_based_prediction = object_prediction(
            process_step, prediction_input_list
        )
        return input_tracking, prediction_input_list, physics_based_prediction


def process_step(x_in: np.ndarray, dt: float) -> np.ndarray:
    """
    Predicts the vehicle state into the future for one step according to dt.

    inputs:
    x_in (type: np.ndarray, shape: (6,)): current state with [x-pos, y-pos, yaw, velocity, yawRate, acceleration]
    dt (type: float): time interval for prediction ahead

    output:
    x_out (type: np.ndarray, shape: (6,)): predicted state with [x-pos, y-pos, yaw, velocity, yawRate, acceleration]

    Note: Yaw-Angle is defined as the angle between the vehicle's heading and global y-axis.
    So yaw = 0.0 means that the vehicle is headed towards global y-axis (north)
    yaw = - pi / 2.0 means that the vehicle is headed towards global x-axis (east)
    """

    # Task:
    # ToDo: Implement the equations for driving straight and for the CTRA-Model.
    # Hints:
    #   - see lecture slides (chapter: Physics-based Prediction)
    #   - keep the if-else-structure to distinguish between straight driving and turning
    ########################
    #  Start of your code  #
    ########################

    input_dim = x_in.shape
    x_out = np.zeros(input_dim)

    if np.abs(x_in[4]) < 1e-3:  # Driving straight
        x_out[0] = x_in[0] - x_in[3] * dt * np.sin(x_in[2])
        x_out[1] = x_in[1] + x_in[3] * dt * np.cos(x_in[2])
    else:  # CTRA-Equations:
        x_out[0] = x_in[0] + x_in[3] / x_in[4] * (np.cos(x_in[4] * dt + x_in[2]) - np.cos(x_in[2]))
        x_out[1] = x_in[1] + x_in[3] / x_in[4] * (np.sin(x_in[4] * dt + x_in[2]) - np.sin(x_in[2]))
        x_out[2] = x_in[2] + x_in[4] * dt
        x_out[3] = x_in[3] + x_in[5] * dt
        x_out[4] = x_in[4]
        x_out[5] = x_in[5]
        pass

    return x_out

    ########################
    #   End of your code   #
    ########################


def plot_results(input_tracking, prediction_input_list, physics_based_prediction):

    ax.cla()

    x_history = input_tracking[:, 0]
    y_history = input_tracking[:, 1]
    ax.plot(x_history, y_history, "r", label="tracking")

    x_now = prediction_input_list[0]
    y_now = prediction_input_list[1]
    ax.plot(x_now, y_now, "kx", label="current position")

    x_fut = physics_based_prediction["x"]
    y_fut = physics_based_prediction["y"]
    ax.plot(x_fut, y_fut, "g:", label="prediction")
    ax.legend()
    ax.set_xlabel("x in m", size=14)
    ax.set_ylabel("y in m", size=14)
    ax.grid(True)
    ax.axis("equal")

    plt.pause(1.0)


if __name__ == "__main__":
    ax = plt.gca()

    physics_model = PhysicsPrediction()
    n_plots = 30

    for _ in range(n_plots):
        (
            input_tracking,
            prediction_input_list,
            physics_based_prediction,
        ) = physics_model.step()
        plot_results(input_tracking, prediction_input_list, physics_based_prediction)
