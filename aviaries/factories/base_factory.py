import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from trajectories import DiscretizedTrajectory
from stable_baselines3.common.vec_env import VecEnv
from aviaries.configuration import Configuration
from gym_pybullet_drones.envs.BaseAviary import BaseAviary


class BaseFactory:

    def __init__(self) -> None:
        pass

    def get_train_env(self) -> VecEnv:
        raise NotImplementedError("Must be overwritten by child class.")

    def get_eval_env(self) -> BaseAviary:
        raise NotImplementedError("Must be overwritten by child class.")

    def get_test_env_gui(self) -> BaseAviary:
        raise NotImplementedError("Must be overwritten by child class.")

    def get_test_env_no_gui(self) -> BaseAviary:
        raise NotImplementedError("Must be overwritten by child class.")
