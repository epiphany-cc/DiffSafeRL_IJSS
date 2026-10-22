import argparse
import numpy as np
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from aviaries.configuration import Configuration
from aviaries.factories.uzh_trajectory_follower_factory import (
    TrajectoryTrackingAviaryFactory
)
import os
from agents.test_policy import run_test
from agents.train_policy import run_train
from typing import Dict
import json
import pickle
from datetime import datetime
from tqdm import tqdm

###### INFRASTRUCTURE PARAMS #######
GUI = False

###### USUALLY NOT CHANGED #########
OBS = ObservationType("kin")  # 'kin' or 'rgb'
ACT = ActionType.TORQUE #ActionType.TORQUE
NUM_DRONES = 1

###### TEST TRAIN FLAGS ############
TRAIN = True
VIS = False  #
TEST = False  #

###### ENVIRONMENT PARAMS ##########
TIMESTEPS = 15000000
N_ENVS = 2
EPISODE_LEN_SEC = 20

pyb_freq = 240
ctrl_freq = 120
DEFAULT_DISCR_LEVEL = EPISODE_LEN_SEC * ctrl_freq
current_time = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

####################################
def save_benchmark(benchmarks: Dict[str, float], file_path: str):
    with open(file_path, "w") as file:
        json.dump(benchmarks, file)

def run(
    gui=GUI,
    timesteps=TIMESTEPS,
    train: bool = TRAIN,
    test: bool = TEST,
    vis: bool = VIS,
    n_envs: int = N_ENVS,
    episode_len_sec: int = EPISODE_LEN_SEC,
    pyb_freq: int = pyb_freq,
    ctrl_freq: int = ctrl_freq,
    discr_level: float = DEFAULT_DISCR_LEVEL
     ):
    path_for_track_test = "../runnables/test_tracks/trajectories_for_DiffRL"

    #output_folder = "../checkpointed_models/model_for_DiffPPO-LSE"

    output_folder = f"../checkpointed_models/{current_time}_model_for_DiffPPO-LSE"

    model_path = "../checkpointed_models/xxxx/best_model.zip"

    TRAJECTORY_TYPES = [
                "random_10m_10s",
                "random",
                "circle",
                "helical",
                "eight",
                "triangle",
                'pentagram'
            ]
    TRAJECTORY_TYPE = TRAJECTORY_TYPES[1]
    eval_set = f"../runnables/test_tracks/trajectories_for_eval/ref_{TRAJECTORY_TYPE}"

    config = Configuration(
        action_type=ACT,
        path_for_track_test=path_for_track_test,
        output_path_location=output_folder,
        n_timesteps=timesteps,
        target_traj=None,
        discr_level=discr_level,
        local=True,
        episode_len_sec=episode_len_sec,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        eval_set=eval_set,
        trajectory_type=TRAJECTORY_TYPE,
        train_use_random=False
    )
    env_factory = TrajectoryTrackingAviaryFactory(
        config=config,
        observation_type=OBS,
        use_gui_for_test_env=gui,
        n_env_training=n_envs,
        seed=0,
    )
    # env_factory = normrlAviaryFactory(
    #     config=config,
    #     observation_type=OBS,
    #     use_gui_for_test_env=gui,
    #     n_env_training=n_envs,
    #     seed=0,
    # )

    if train:
        run_train(config=config, env_factory=env_factory, model_load_path=None)

    if vis:
        run_test(config=config, env_factory=env_factory, current_time=current_time)

    if test:
        tracking_files_set = [f for f in os.listdir(eval_set) if f.endswith(".npy")]
        tracking_files_set.sort(key=lambda f: int(f.split('.')[0]))
        print(f"{len(tracking_files_set)} tracking")

        for i in tqdm(range(len(tracking_files_set)), desc="", unit="file"):
            tracking_file_path = os.path.join(eval_set, tracking_files_set[i])
            track_traj = np.load(tracking_file_path, allow_pickle=True)
            config.update_trajectory(track_traj)
            env_factory.set_config(config)

            run_test(
                config=config,
                env_factory=env_factory,
                eval_mode=True,
                nth_trajs=i,
                current_time=current_time
            )


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description="Single agent reinforcement learning example script")
    parser.add_argument("--gui", default=GUI, type=str2bool, help="Whether to use PyBullet GUI (default: True)", metavar="",)
    parser.add_argument("--timesteps", default=TIMESTEPS, type=int, help="number of train timesteps before stopping", metavar="",)
    parser.add_argument("--train", default=TRAIN, type=str2bool, help="Whether to train (default: True)", metavar="",)
    parser.add_argument("--test", default=TEST, type=str2bool, help="Whether to test (default: True)", metavar="",)
    parser.add_argument("--vis", default=VIS, type=str2bool, help="Whether to visualise learned policy (default: True)", metavar="",)
    parser.add_argument("--n_envs", default=N_ENVS, type=int, help="number of parallel environments", metavar="",)
    parser.add_argument("--episode_len_sec", default=EPISODE_LEN_SEC, type=int, help="number of parallel environments", metavar="",)
    ARGS = parser.parse_args()

    run(**vars(ARGS))
