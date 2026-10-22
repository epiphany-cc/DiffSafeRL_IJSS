import sys
sys.path.append("../..")
from trajectories import TrajectoryFactory
import numpy as np
import os
import shutil
from datetime import datetime
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Get_trajs():

    def __init__(self, base_folder):  #TODO
        self.folder = base_folder

    def get_eval_trajs(self, n_discr, ctrl_pyb, take_off_discr, trajectory_type, num_trajs=1, is_save_track_traj=False, is_save_plot_traj=False):  # todo

        datetime_string = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        if is_save_track_traj:
            track_target_folder = (
                    os.path.join(self.folder, trajectory_type)
                    + f"_{datetime_string}"
            )
            os.makedirs(track_target_folder, exist_ok=True)
        elif is_save_plot_traj:
            plot_target_folder = (
                    os.path.join(self.folder, trajectory_type)
                    + f"_for_tracking_{datetime_string}"
            )
            os.makedirs(plot_target_folder, exist_ok=True)


        for i in range(num_trajs):
            if trajectory_type == 'random':
                traj = TrajectoryFactory.get_random_trajectory(
                    start=np.array([0, 0, 2]),
                    n_discr_level=n_discr,
                    n_ctrl_points=10,
                    std_dev_deg=50,
                    distance_between_ctrl_points=1,
                    init_dir=None,
                    return_ctrl_points=False
                )

            elif trajectory_type == 'circle':
                traj = TrajectoryFactory.get_circular_trajectory(
                    center=np.array([0, 0, 2]),
                    radius=1,
                    n_ctrl_points=n_discr,
                    n_discr_level=n_discr,
                    start_angle=0.0,
                    end_angle=2 * math.pi,
                    return_ctrl_points=False
                )

            elif trajectory_type == 'helical':
                traj = TrajectoryFactory.get_helical_trajectory(
                    start=np.array([0, 0, 2]),
                    radius=1,
                    height_per_turn=1,
                    num_turns=2,
                    points_per_turn=n_discr/2,
                    n_discr_level=n_discr,
                    return_ctrl_points=False
                )

            elif trajectory_type == 'eight':
                traj = TrajectoryFactory.get_figure_eight_trajectory(
                    center=np.array([0, 0, 2]),
                    major_axis=2.0,
                    minor_axis=1,
                    n_ctrl_points=int(n_discr),
                    n_discr_level=int(n_discr),
                    return_ctrl_points=False
                )
            elif trajectory_type == 'pentagram':
                traj = TrajectoryFactory.get_pentagram_trajectory(
                    center=np.array([0, 0, 2]),
                    radius=1.0,
                    n_ctrl_points_per_side=int(n_discr/5),
                    n_discr_level=int(n_discr)
                )

            elif trajectory_type == 'triangle':
                traj = TrajectoryFactory.get_triangle_trajectory(
                    center=np.array([0, 0, 2]),
                    side_length=2.0,
                    n_ctrl_points_per_side=n_discr/3,
                    n_discr_level=n_discr,
                    return_ctrl_points=False
                )

            np_refer_traj = (traj[:].coordinate).T  # numpy (n_discr,3)
            refer_traj_first_point = np_refer_traj[0]  # shape=(3,)
            refer_traj_last_point = np_refer_traj[-1]  # shape=(3,)
            take_off_point = refer_traj_first_point.copy()
            landing_point = refer_traj_last_point.copy()
            take_off_point[2] = 0
            landing_point[2] = 0
            if take_off_discr != 0:
                np_take_off_traj = self.get_takeoff_landing_trajs(take_off_point=take_off_point,
                                                                  refer_traj_first_point=refer_traj_first_point,
                                                                  refer_traj_last_point=refer_traj_last_point,
                                                                  landing_point=landing_point,
                                                                  n_ctrl=take_off_discr,
                                                                  n_discr=take_off_discr,
                                                                  hover_count=5
                                                                  )
                np_all_traj = np.concatenate([np_take_off_traj, np_refer_traj], axis=0)
                target_num_points = len(np_all_traj)
                # ====================================================

                diffs = np.diff(np_all_traj, axis=0)
                distances = np.linalg.norm(diffs, axis=1)

                cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
                total_length = cumulative_dist[-1]

                _, unique_indices = np.unique(cumulative_dist, return_index=True)
                unique_indices = np.sort(unique_indices)

                cumulative_dist_clean = cumulative_dist[unique_indices]
                np_all_traj_clean = np_all_traj[unique_indices]

                uniform_dist = np.linspace(0, total_length, target_num_points)

                interpolator = interp1d(cumulative_dist_clean, np_all_traj_clean, axis=0, kind='linear')
                np_all_traj_uniform = interpolator(uniform_dist)
                if is_save_track_traj:
                    traj_folder = os.path.join(track_target_folder, f"{i}.npy")
                    np.save(traj_folder, np_all_traj_uniform)
                if is_save_plot_traj:
                    traj_folder = os.path.join(plot_target_folder, f"{i}.npy")
                    np.save(traj_folder, np_refer_traj)

            else:
                if is_save_plot_traj:
                    traj_folder = os.path.join(plot_target_folder, f"{i}.npy")
                    np.save(traj_folder, np_refer_traj)

        if take_off_discr != 0:
            return np_refer_traj, np_all_traj_uniform
        else:
            return np_refer_traj

    def get_takeoff_landing_trajs(self,
                          take_off_point,
                          refer_traj_first_point,
                          refer_traj_last_point,
                          landing_point,
                          n_ctrl,
                          n_discr,
                          hover_count,
                          ):

        np_take_off_traj = TrajectoryFactory.get_linear_trajectory(
            take_off_first_point=take_off_point,
            take_off_last_point=refer_traj_first_point,
            landing_first_point=refer_traj_last_point,
            landing_last_point=landing_point,
            n_ctrl_points=n_ctrl,
            n_discr_level=n_discr,
            hover_count=hover_count,
            take_off_only=True
        )
        return np_take_off_traj


if __name__ == '__main__':
    base_folder: str = "../test_tracks/trajectories_for_eval"
    TRAJECTORY_TYPES = [
        "random",  #
        "circle",  #
        "helical",  #
        "eight",  #
        "triangle",  #
        'pentagram'
    ]
    TRAJECTORY_TYPE = TRAJECTORY_TYPES[0]
    Get_trajs = Get_trajs(base_folder=base_folder)
    plot_trajs = Get_trajs.get_eval_trajs(num_trajs=100,
                                          ctrl_pyb=120,
                                          n_discr=2400,
                                          take_off_discr=0,
                                          is_save_track_traj=False,
                                          is_save_plot_traj=True,
                                          trajectory_type=TRAJECTORY_TYPE
                                          )

