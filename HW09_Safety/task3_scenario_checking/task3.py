"""
Goal of Task 3:
    The goal is to bring the information of the practical session and the homework together to sort out the four unsafe
    trajectories among the five given trajectories in the practice lecture.

Hint:
    In order to properly run the code in this homework, additional packages have to be installed via pip.
    Use:
        `pip install trajectory-planning-helpers`
        `pip install scenario-testing-tools`
"""


import numpy as np
import shapely.geometry
import practice_session


def check_trajectory(ttc, a_comb, bound_collision, ttc_treshold=2.0, a_comb_threshold=15.0):
    """
    inputs:
        ttc (type: float): time to collision (TTC) [in s]
        a_comb (type: np.ndarray): combined acceleration acting on the tires along the trajectory [in m/s^2]
        bound_collision (type: bool): states whether a collision with a bound was detected (True if trajectory collides)
        ttc_treshold (type: float): (optional) a TTC below or equal to this value is considered unsafe [in s]
        a_comb_threshold (type: float): (optional) a combined acceleration above this value is considered unsafe
                                        [in m/s^2]

    output:
        safe (type: bool): flag indicating the safety for the given trajectory (True for safe trajectory)
    """

    # Subtask 1:
    # ToDo: Implement a function that states whether a trajectory is safe based on the given metrics TTC
    #  (practice session), maximum combined acceleration (practice session) and a flag for bound collisions (homework).
    # Hint: The thresholds for the floating point values are given as default parameters.
    ########################
    #  Start of your code  #
    ########################

    safe_bc = not bound_collision

    safe_acc = True
    for j in range(len(a_comb)):
        if a_comb[j] > a_comb_threshold:
            safe_acc = False
            break

    safe_ttc = (ttc > ttc_treshold)

    if safe_bc and safe_acc and safe_ttc:
        safe = True
    else:
        safe = False

    ########################
    #   End of your code   #
    ########################

    return safe


def check_bound_collision(path, bound, veh_width=2.8):
    """
    Identifies if a provided path collides with a track boundary. Both, the path and the boundary, are provided as a
    sequence of coordinates connected by straights. This function can then be used to detect collisions with any of
    the track boundaries.

    inputs:
        path (type: np.ndarray): path of the ego trajectory with columns x, y [in m]
        bound (type: np.ndarray): track boundary with columns x, y [in m]
        veh_width (type: float): (optional) vehicle width [in m]

    output:
        intersection (type: bool): flag indicating intersection of the two provided paths (True if intersection exists)
    """

    # Subtask 2:
    # ToDo: Insert code of Task 2. Delete "raise ValueError" statement afterwards.
    ########################
    #  Start of your code  #
    ########################

    """ -- SOLUTION OF TASK 2 / REQUIRED FOR DEVELOPMENT, NOT FOR PASSING THIS TASK -- """
    path_bound = shapely.geometry.LineString(path).buffer(veh_width / 2)
    bound_obj = shapely.geometry.LineString(bound)
    intersection = path_bound.intersects(bound_obj)

    ########################
    #   End of your code   #
    ########################

    return intersection


if __name__ == "__main__":
    # init data from practice session
    data = practice_session.Data()

    # find index in trajectory that is closest to t = 4s
    t = 4.0
    idx_t = np.argmin(np.abs(data.traj_ego1[:, 0] - t))

    # for all ego trajectory candidates
    safe = []
    for i, traj_ego in enumerate([data.traj_ego1, data.traj_ego2, data.traj_ego3, data.traj_ego4, data.traj_ego5]):
        # calculate ttc
        ttc = practice_session.calc_ttc(pos_ego=traj_ego[idx_t, 1:3],
                                        vel_ego=traj_ego[idx_t, 5],
                                        pos_obj=data.traj_obj[idx_t, 1:3],
                                        vel_obj=data.traj_obj[idx_t, 4])

        # calculate combined acceleration
        a_comb = practice_session.calc_a_comb(traj_ego=traj_ego)

        # check for boundary collision
        check_left = check_bound_collision(path=traj_ego[:, 1:3], bound=data.bound_l)
        check_right = check_bound_collision(path=traj_ego[:, 1:3], bound=data.bound_r)
        bound_collision = (check_left or check_right)

        # evaluate safety
        safe.append(check_trajectory(ttc=ttc,
                                     a_comb=a_comb,
                                     bound_collision=bound_collision))

        if safe[i]:
            print("Trajectory %i is rated: SAFE " % (i + 1))
        else:
            print("Trajectory %i is rated: UNSAFE " % (i + 1))

    # assert if rating is not the expected one
    assert all([a == b for a, b in zip(safe, [False, True, False, False, False])])
