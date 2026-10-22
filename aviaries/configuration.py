import numpy as np
from gym_pybullet_drones.utils.enums import ActionType
from trajectories import DiscretizedTrajectory
import numpy as np
from typing import List


class Configuration:
    action_type: ActionType
    plot_track_trajectory: DiscretizedTrajectory
    target_reward: float
    output_path_location: str
    n_timesteps: int
    local: bool
    episode_len_sec: float
    waypoint_buffer_size: int

    def __init__(
        self,
        action_type: ActionType,
        output_path_location: str,
        path_for_track_test: str,
        eval_set: str,
        trajectory_type: str,
        n_timesteps: int,
        local: bool,
        episode_len_sec: int = 10,
        waypoint_buffer_size: int = 2,
        pyb_freq: int = 240,
        ctrl_freq: int = 240,
        discr_level: int = 960,
        target_traj=None,
        t_reward=None,
        train_use_random=False

    ) -> None:
        self.trajectory_type = trajectory_type
        self.eval_set = eval_set
        self.action_type = action_type
        self.target_traj = target_traj
        if t_reward is None:
            self.target_reward = float("inf")
        else:
            t_reward = t_reward
        self.output_path_location = output_path_location
        self.path_for_track_test = path_for_track_test
        self.n_timesteps = n_timesteps
        self.local = local
        self.episode_len_sec = episode_len_sec
        self.waypoint_buffer_size = waypoint_buffer_size
        self.pyb_freq = pyb_freq
        self.ctrl_freq = ctrl_freq
        self.discr_level = discr_level
        self.train_use_random = train_use_random

    def update_trajectory(self, target_traj):
        self.target_traj = target_traj
