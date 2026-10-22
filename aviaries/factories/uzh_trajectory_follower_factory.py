import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import DiscretizedTrajectory
from stable_baselines3.common.vec_env import VecEnv
from aviaries.configuration import Configuration
from aviaries.UZHAviary import UZHAviary
from .base_factory import BaseFactory

class TrajectoryTrackingAviaryFactory(BaseFactory):
    action_type: ActionType
    observation_type: ObservationType
    target_traj: DiscretizedTrajectory
    n_env_training: int
    seed: int
    use_gui_for_test_env: bool
    output_path_location: str
    episode_len_sec: int
    def __init__(
        self,
        config: Configuration,
        observation_type: ObservationType,
        use_gui_for_test_env: bool = True,
        n_env_training: int = 20,
        seed: int = 0,
        eval_mode=False,
    ) -> None:
        super().__init__()
        self.eval_mode = eval_mode
        self.observation_type = observation_type
        self.n_env_training = n_env_training
        self.seed = seed
        self.use_gui_for_test_env = use_gui_for_test_env
        self.set_config(config)

    def set_config(self, config: Configuration):
        self.action_type = config.action_type
        self.pyb_freq = config.pyb_freq
        self.ctrl_freq = config.ctrl_freq
        self.discr_level = config.discr_level
        self.target_traj = config.target_traj
        self.eval_set = config.eval_set
        self.trajectory_type = config.trajectory_type
        self.episode_len_sec = config.episode_len_sec

    def get_train_env(self) -> VecEnv:
        train_env = make_vec_env(
            UZHAviary,
            env_kwargs=dict(
                target_trajectory=self.target_traj,
                obs=self.observation_type,
                act=self.action_type,
                pyb_freq=self.pyb_freq,
                ctrl_freq=self.ctrl_freq,
                discr_level=self.discr_level,
                episode_len_sec=self.episode_len_sec,
                trajectory_type=self.trajectory_type
            ),
            n_envs=self.n_env_training,
            seed=self.seed,
        )

        return train_env

    def get_eval_env(self):
        eval_env = UZHAviary(
            target_trajectory=self.target_traj,
            obs=self.observation_type,
            act=self.action_type,
            pyb_freq=self.pyb_freq,
            ctrl_freq=self.ctrl_freq,
            discr_level=self.discr_level,
            episode_len_sec=self.episode_len_sec,
            eval_mode=self.eval_mode,
            log_positions=True if self.eval_mode else False,
            eval_set=self.eval_set,
            trajectory_type=self.trajectory_type
        )
        return eval_env

    def get_test_env_gui(self):
        test_env = UZHAviary(
            target_trajectory=self.target_traj,
            gui=self.use_gui_for_test_env,
            obs=self.observation_type,
            act=self.action_type,
            pyb_freq=self.pyb_freq,
            ctrl_freq=self.ctrl_freq,
            discr_level=self.discr_level,
            record=False,
            episode_len_sec=self.episode_len_sec,
            eval_mode=self.eval_mode,
            log_positions=True if self.eval_mode else False,
            eval_set=self.eval_set,
            trajectory_type=self.trajectory_type
        )
        return test_env

    def get_test_env_no_gui(self):
        test_env_nogui = UZHAviary(
            target_trajectory=self.target_traj,
            gui=False,
            obs=self.observation_type,
            act=self.action_type,
            pyb_freq=self.pyb_freq,
            ctrl_freq=self.ctrl_freq,
            discr_level=self.discr_level,
            record=False,
            episode_len_sec=self.episode_len_sec,
            eval_mode=self.eval_mode,
            log_positions=True if self.eval_mode else False,
            eval_set=self.eval_set,
            trajectory_type=self.trajectory_type
        )
        return test_env_nogui


