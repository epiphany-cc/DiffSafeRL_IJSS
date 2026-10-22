"""
Methods to test a policy.
"""

from aviaries.factories.base_factory import BaseFactory
from aviaries.configuration import Configuration
import os
from stable_baselines3.common.evaluation import evaluate_policy
from aviaries.UZHAviary import UZHAviary
from gym_pybullet_drones.utils.Logger import Logger
from stable_baselines3 import TD3, PPO
from agents.DiffPPO_policy import SafePPOPolicy
import time
import numpy as np
from gym_pybullet_drones.utils.utils import sync


def test_simple_follower(
    model_path: str,
    test_env: UZHAviary,
    path_for_track_test: str,
    trajectory_type: str,
    eval_mode=False,
    nth_trajs=0,
    current_time=None
):

    # load model
    if os.path.isfile(model_path + "/best_model.zip"):
        path = model_path + "/best_model.zip"
        #model = TD3.load(path, custom_objects={'policy_class': SafeTD3Policy})
        model = PPO.load(path, env=test_env)
    else:
        print("[ERROR]: no model under the specified path", model_path)

    # visualise in test environment
    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=path_for_track_test,
        colab=False,
        trajectory_type=trajectory_type,
        datetime=current_time
    )

    obs, info = test_env.reset(seed=42, options={})
    test_env.reset_vars()
    start = time.time()
    range_steps = (test_env.EPISODE_LEN_SEC) * test_env.CTRL_FREQ
    for i in range(range_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        drone_state = test_env._getDroneStateVector(nth_drone=0) #(20,)
        #min_cbf_value = model.actor.current_cbf_info  # TD3,SAC
        min_cbf_value = model.policy.current_cbf_info  # PPO
        logger.log(
            nth_drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=drone_state,
            min_cbf_value=min_cbf_value
        )
        if not eval_mode:
            sync(i, start, test_env.CTRL_TIMESTEP)
            #print(f'terminated-gui-gui-gui:{terminated}')

        if terminated or truncated:
            if eval_mode:
                logger.save(nth_trajs)
                break
            break
    test_env.close()

def run_test(config: Configuration, env_factory: BaseFactory, eval_mode=False, nth_trajs=0, current_time=None):

    if not eval_mode:
        test_env = env_factory.get_test_env_gui()
    else:
        test_env = env_factory.get_test_env_no_gui()

    test_simple_follower(
        model_path=config.output_path_location,
        test_env=test_env,
        path_for_track_test=config.path_for_track_test,
        eval_mode=eval_mode,
        trajectory_type=config.trajectory_type,
        nth_trajs=nth_trajs,
        current_time=current_time
    )

