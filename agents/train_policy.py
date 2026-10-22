"""
Methods to learn a policy.
"""
import time
import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from aviaries.configuration import Configuration
from aviaries.factories.base_factory import BaseFactory
import os
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from stable_baselines3.common.noise import NormalActionNoise
from agents.DiffPPO_policy import SafePPOPolicy
'''---------------------------------------------------------------------------------------------
class RewardBreakdownCallback():

---------------------------------------------------------------------------------------------
class HOCBFActivationCallback():
callback 'enable_hocbfs'=True hocbf
---------------------------------------------------------------------------------------------'''
class RewardBreakdownCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.key_to_tag = {
            "Reward of tracking": "Rewards/tracking",
            "Penalty for safe intervention": "Rewards/safe_intervention",
            "Raw safe intervention": "Rewards/raw_intervention",

            "Safe action abs/u_T": "ActionAbs/Safe/u_T",
            "Safe action abs/tau_x": "ActionAbs/Safe/tau_x",
            "Safe action abs/tau_y": "ActionAbs/Safe/tau_y",
            "Safe action abs/tau_z": "ActionAbs/Safe/tau_z",

            "RL action abs/u_T": "ActionAbs/RL/u_T",
            "RL action abs/tau_x": "ActionAbs/RL/tau_x",
            "RL action abs/tau_y": "ActionAbs/RL/tau_y",
            "RL action abs/tau_z": "ActionAbs/RL/tau_z",

            "Safe action saturation": "Saturation/safe_action",
            "RL action saturation": "Saturation/rl_action",

        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for key, tag in self.key_to_tag.items():
            values = []

            for info in infos:
                if key in info and info[key] is not None:
                    values.append(info[key])

            if values:
                mean_val = float(np.mean(values))
                self.logger.record(tag, mean_val)

        return True


class ATTSCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(ATTSCallback, self).__init__(verbose)
        self.step_times = []
        self.last_time = None

    def _on_rollout_start(self) -> None:
        self.last_time = time.perf_counter()

    def _on_step(self) -> bool:
        current_time = time.perf_counter()
        if self.last_time is not None:
            self.step_times.append(current_time - self.last_time)
        self.last_time = current_time
        return True

    def _on_rollout_end(self) -> None:
        if len(self.step_times) > 0:
            atts = np.mean(self.step_times)

            self.logger.record("time/ATTS_seconds", atts)

            self.step_times = []

class CBFInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        policy = self.model.policy

        if hasattr(policy, "current_cbf_info"):
            cbf_info = policy.current_cbf_info

            h0 = cbf_info.get("h0", [0.0])[0]
            h1 = cbf_info.get("h1", [0.0])[0]
            h2 = cbf_info.get("h2", [0.0])[0]

            self.logger.record("CBF/h0", h0)
            self.logger.record("CBF/h1", h1)
            self.logger.record("CBF/h2", h2)
            self.logger.record("CBF/active", float(h2 < 0.0))

        return True

def run_train(config: Configuration, env_factory: BaseFactory, model_load_path: str = None):

    ################################# CONFIG ###################################
    train_env = env_factory.get_train_env()
    eval_env = env_factory.get_eval_env()
    ################################### model ##################################

    if model_load_path and os.path.exists(model_load_path):
        print(f"[INFO] Loading existing model from {model_load_path} for continuing training...")


    else:
        #todo#### PPO ######

        policy_kwargs = dict(activation_fn=nn.Tanh,
                             net_arch=dict(pi=[256, 256], vf=[256, 256]),
                             log_std_init=-1.0
                             )  # log_std_init = -1.0
        model = PPO(
            SafePPOPolicy,
            train_env,
            batch_size=4096,
            learning_rate=0.0003,
            tensorboard_log=config.output_path_location + "/tb/",
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda",
        )

    # callbacks
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=2350, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=config.output_path_location + "/",
        log_path=config.output_path_location + "/",
        eval_freq=int(2000),
        deterministic=True,
        render=False,
    )

    # todo callback info for tensorboard
    various_rewards_callback = RewardBreakdownCallback()
    atts_callback = ATTSCallback()
    CBFInfo = CBFInfoCallback()
    # ##########################################################

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)
    print("[INFO] Number of timesteps:", config.n_timesteps)

    # TRAIN ####################################################
    callback_list = [eval_callback, various_rewards_callback, atts_callback, CBFInfo]
    #callback_list = [eval_callback]
    model.learn(total_timesteps=config.n_timesteps,
                callback=callback_list,
                log_interval=100,
                reset_num_timesteps=(not bool(model_load_path))
                )

    # save model
    model.save(config.output_path_location + "/final_model.zip")
    print(config.output_path_location)

    # print training progression
    with np.load(config.output_path_location + "/evaluations.npz") as data:
        for j in range(data["timesteps"].shape[0]):
            print(str(data["timesteps"][j]) + "," + str(data["results"][j][0]))

    # ##########################################################