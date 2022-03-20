import numpy as np
import scenario_testing_tools as stt


class Data(object):

    def __init__(self):
        # Scenario files
        file_scn1 = "scenarios/scenario_1.saa"
        file_scn2 = "scenarios/scenario_2.saa"
        file_scn3 = "scenarios/scenario_3.saa"
        file_scn4 = "scenarios/scenario_4.saa"
        file_scn5 = "scenarios/scenario_5.saa"

        # -- get track boundaries from scenario file --
        self.bound_l, self.bound_r = stt.get_scene_track.get_scene_track(file_path=file_scn1)

        # -- trajectory of object vehicle --
        # init list containers (length not known beforehand)
        t_obj = []
        x_obj = []
        y_obj = []
        psi_obj = []
        v_obj = []

        # get all entries from scenario file
        i = 0
        while True:
            try:
                # get object list entry, if requested time stamp is not in file an error is raised
                data = stt.get_scene_timesample.get_scene_timesample(file_path=file_scn1,
                                                                     t_in=i)
                time = data[0]
                obj_list = data[8]

                # get first (and only) object in the object list
                obj = next(iter(obj_list.values()))

                # append data to list containers
                t_obj.append(time)
                x_obj.append(obj['X'])
                y_obj.append(obj['Y'])
                psi_obj.append(obj['psi'])
                v_obj.append(obj['vel'])

                # increase counter
                i += 1
            except:
                break

        # get numpy array form list containers, columns [t, x, y, psi, v]
        self.traj_obj = np.column_stack((t_obj, x_obj, y_obj, psi_obj, v_obj))

        # -- trajectories of ego-vehicle --
        # get first ego trajectory data
        data = stt.get_scene_ego_traj.get_scene_ego_traj(file_path=file_scn1,
                                                         append_plan=False)

        # get numpy array for first ego trajectory with columns [t, x, y, psi, curv, v, a]
        self.traj_ego1 = np.column_stack((data[:7]))

        # execute this for the remaining four trajectories (in line)
        self.traj_ego2 = np.column_stack((stt.get_scene_ego_traj.get_scene_ego_traj(file_path=file_scn2,
                                                                                    append_plan=False)[:7]))
        self.traj_ego3 = np.column_stack((stt.get_scene_ego_traj.get_scene_ego_traj(file_path=file_scn3,
                                                                                    append_plan=False)[:7]))
        self.traj_ego4 = np.column_stack((stt.get_scene_ego_traj.get_scene_ego_traj(file_path=file_scn4,
                                                                                    append_plan=False)[:7]))
        self.traj_ego5 = np.column_stack((stt.get_scene_ego_traj.get_scene_ego_traj(file_path=file_scn5,
                                                                                    append_plan=False)[:7]))


def calc_ttc(pos_ego, vel_ego, pos_obj, vel_obj, veh_len=4.7):
    """
    Calculates the time to collision (TTC) for a given ego vehicle (pos, vel) and an object vehicle (pos, vel).
    Assumption: the provided object vehicle is _in front_ of the ego vehicle.

    inputs:
        pos_ego (type: np.ndarray): position of ego vehicle as numpy array with columns x, y [in m]
        vel_ego (type: float): velocity of ego vehicle [in m/s]
        pos_obj (type: np.ndarray): position of object vehicle as numpy array with columns x, y [in m]
        vel_obj (type: float): velocity of object vehicle [in m/s]
        veh_len (type: float): (optional) vehicle length (assumed identical for both) [in m]

    output:
        ttc (type: np.float64): time to collision [in s]
    """

    # check if ego vehicle is faster than leading vehicle (otherwise it will never reach the other vehicle)
    if vel_ego > vel_obj:
        # calculate distance between vehicles (bumper to bumper)
        dist = np.hypot(pos_ego[0] - pos_obj[0], pos_ego[1] - pos_obj[1]) - veh_len

        # calculate ttc
        ttc = dist / (vel_ego - vel_obj)

    else:
        ttc = np.inf

    return ttc


def calc_a_comb(traj_ego, c_drag=0.954, m_veh=1160.0):
    """
    Calculates the combined acceleration acting on the tires for a given trajectory.

    inputs:
        traj_ego (type: np.ndarray): ego trajectory with columns [t, x, y, psi, curv, v, a]
        c_drag (type: float): (optional) vehicle specific drag coefficient
        m_veh (type: float): (optional) vehicle mass

    output:
        a_comb (type: np.ndarray): combined acceleration acting on the tires along the trajectory
    """

    # for each point on the planed trajectory, extract curvature, velocity and longitudinal acceleration
    ego_curve = traj_ego[:, 4]
    ego_velocity = traj_ego[:, 5]
    a_lon_used = traj_ego[:, 6]

    # for each point on the planned trajectory, calculate the lateral acceleration based on curvature and velocity
    a_lat_used = np.power(ego_velocity[:], 2) * ego_curve[:]

    # calculate equivalent longitudinal acceleration of drag force along velocity profile
    a_lon_drag = np.power(ego_velocity[:], 2) * c_drag / m_veh

    # drag reduces requested deceleration but increases requested acceleration at the tire
    a_lon_used += a_lon_drag

    # calculate used combined acceleration
    a_comb = np.sqrt(np.power(np.abs(a_lon_used), 2) + np.power(np.abs(a_lat_used), 2))

    return a_comb
