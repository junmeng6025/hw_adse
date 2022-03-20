"""
Goal of Task 3:
    Implement the necessary equations to run a Extended Kalman Filter (EKF).

Hint: Execute EKF.py to check your implementation.
"""


import numpy as np


def ProjectErrorCovariance(JA, P, Q):
    """
    inputs:
        JA (type: np.matrix)
        P (type: np.ndarray)
        Q (type: np.ndarray)

    output:
        P (type: np.matrix)
    """

    # Subtask 1:
    # ToDo: Project the error covariance ahead.
    ########################
    #  Start of your code  #
    ########################
    P = JA * P * JA.T + Q
    ########################
    #   End of your code   #
    ########################

    return P


def ComputeKalmanGain(JH, P, R):
    """
    inputs:
        JH (type: np.matrix)
        P (type: np.matrix)
        R (type: np.ndarray)

    output:
        K (type: np.matrix)
    """

    # Subtask 2:
    # ToDo: Compute the Kalman Gain.
    # Hints:
    #   - you can use intermediate terms to make things easier
    #   - use np.linalg.inv()
    ########################
    #  Start of your code  #
    ########################
    K = (P * JH.T) * np.linalg.inv(JH * P * JH.T + R)
    ########################
    #   End of your code   #
    ########################

    return K


def UpdateErrorCovariance(P, EYE, K, JH):
    """
    inputs:
        P (type: np.matrix)
        EYE (type: np.ndarray)
        K (type: np.matrix)
        JH (type: np.matrix)

    output:
        P (type: np.matrix)
    """

    # Subtask 3:
    # ToDo: Update the error covariance.
    ########################
    #  Start of your code  #
    ########################
    P = (EYE - (K * JH)) * P
    ########################
    #   End of your code   #
    ########################

    return P
