from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
    ImageType
)

from scipy.interpolate import CubicSpline
import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from runnables.utils.get_eval_trajectories import Get_trajs
from quadprog import solve_qp
from trajectories import TrajectoryFactory
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
txt_colour = [0, 0, 0]
txt_size = 1.0

dummy_text = lambda txt, pos, client_id: p.addUserDebugText(
    txt,
    pos,
    lifeTime=0,
    textSize=txt_size,
    textColorRGB=txt_colour,
    physicsClientId=client_id,
)

refreshed_text = lambda txt, pos, client_id, replace_id: p.addUserDebugText(
    txt,
    pos,
    lifeTime=0,
    textSize=txt_size,
    textColorRGB=txt_colour,
    physicsClientId=client_id,
    replaceItemUniqueId=replace_id,
)

class UZHAviary(BaseAviary):
    ###############################################################################
    def __init__(
        self,
        target_trajectory=None,
        drone_model: DroneModel = DroneModel.CF2X,
        initial_rpys=np.array([[0, 0, 0]]),
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 60,
        discr_level: int = 1200,
        gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.TORQUE,
        episode_len_sec: int = 20,
        eval_mode=False,
        log_positions=False,
        eval_set: str='',
        trajectory_type: str=''
    ):
        self.trajectory_type = trajectory_type
        self.get_eval_trajs = Get_trajs(base_folder=eval_set)
        vision_attributes = True if obs == ObservationType.RGB else False
        self.obstacles = True

        self.r_track = 0.0
        self.r_safe = 0.0
        self.raw_intervention = 0.0

        self.safe_u_T_abs = 0.0
        self.safe_tau_x_abs = 0.0
        self.safe_tau_y_abs = 0.0
        self.safe_tau_z_abs = 0.0

        self.rl_u_T_abs = 0.0
        self.rl_tau_x_abs = 0.0
        self.rl_tau_y_abs = 0.0
        self.rl_tau_z_abs = 0.0

        self.safe_action_saturation = 0.0
        self.rl_action_saturation = 0.0

        self.EPISODE_LEN_SEC = episode_len_sec
        self.ctrl_freq = ctrl_freq
        self.NUM_DRONES = 1
        self.discr_level = discr_level
        # FOR DEVELOPMENT
        self.target_trajectory = target_trajectory
        # FOR EVAL
        self.eval_mode = eval_mode
        self.GUI = gui
        self.trajectory = self.set_trajectory()  # (self.discr_level,3)

        self.ACT_TYPE = act
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.ATTITUDE_PID,
                   ActionType.RL_ATT_PID, ActionType.TORQUE]:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(self.NUM_DRONES)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        self.current_action = None
        self.INIT_XYZS = self.trajectory[0:1, :] # (1,3)

        super().__init__(
            drone_model=drone_model,
            num_drones=self.NUM_DRONES,
            initial_xyzs=self.INIT_XYZS,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=self.ctrl_freq,
            gui=gui,
            record=record,
            obstacles=self.obstacles,  # Add obstacles for RGB observations and/or FlyThruGate
            user_debug_gui=False,  # Remove of RPM sliders from all single agent learning aviaries
            vision_attributes=vision_attributes,
            log_positions=log_positions
        )

        # Visualisation
        self.visualised = False
        pos_x, pos_y, pos_z = self.INIT_XYZS[0]
        init_text_pos = [float(pos_x), float(pos_y), float(pos_z)]
        self.text_id = dummy_text(f"Track error: {0:.3f}", init_text_pos, self.CLIENT)

    def reset_vars(self):
        self.trajectory = self.set_trajectory()

    def set_trajectory(self):
        if self.GUI:
            n_discr = int(self.discr_level)
            trajectory = self.get_eval_trajs.get_eval_trajs(n_discr=n_discr,
                                                            ctrl_pyb=self.ctrl_freq,
                                                            take_off_discr=0,
                                                            trajectory_type=self.trajectory_type,
                                                            is_save_track_traj=False,
                                                            is_save_plot_traj=False,
                                                            )   # (n,3)
        elif (not self.GUI) and (self.target_trajectory is not None):
            trajectory = self.target_trajectory

        else:
            # ctrl_wps = TrajectoryFactory.get_random_trajectory(
            #     start=np.array([0, 0, 2]),
            #     n_discr_level=self.discr_level,
            #     n_ctrl_points=10,
            #     std_dev_deg=np.random.choice([15, 25, 30]),
            #     distance_between_ctrl_points=1,
            #     init_dir=None,
            #     return_ctrl_points=False,
            # )
            # trajectory = np.array([x.coordinate for x in ctrl_wps])
            TRAJECTORY_TYPES = [
                "random",
                "circle",
                "helical",
                "eight",
                "triangle",
                'pentagram'
            ]
            num = np.random.randint(1, 6)
            trajectory = self.get_eval_trajs.get_eval_trajs(n_discr=self.discr_level,
                                                            ctrl_pyb=self.ctrl_freq,
                                                            take_off_discr=0,
                                                            trajectory_type=TRAJECTORY_TYPES[num],
                                                            is_save_track_traj=False,
                                                            is_save_plot_traj=False,
                                                            )



        return trajectory

    def _actionSpace(self):
        """Returns the action space of the environment.
        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.
        """
        # if self.ACT_TYPE in [ActionType.TORQUE]:
        #     size = 8
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL, ActionType.ATTITUDE_PID, ActionType.TORQUE]:
            size = 8
        elif self.ACT_TYPE in [ActionType.PID, ActionType.RL_ATT_PID]:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseAviary._actionSpace()")
            exit()
        act_lower_bound = np.full((self.NUM_DRONES, size), -1.0)
        act_upper_bound = np.full((self.NUM_DRONES, size), 1.0)

        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        # Position [0:3]  # Orientation [3:7]  # Roll, Pitch, Yaw [7:10]  # Linear Velocity [10:13]  # Angular Velocity [13:16]
        lo = -np.inf
        hi = np.inf
        track_state_num = 22
        static_obs, dynamic_obs = self._get_obstacle_pos(num_obstacles=3)
        obs_num = len(static_obs) * 3
        obs_size = track_state_num + obs_num

        obs_lower_bound = np.full((self.NUM_DRONES, obs_size), lo)
        obs_upper_bound = np.full((self.NUM_DRONES, obs_size), hi)

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        state = self._getDroneStateVector(0)

        total_len_traj = int(max(self.trajectory.shape[0], self.trajectory.shape[1]))
        static_obs, dynamic_obs = self._get_obstacle_pos(num_obstacles=3)
        safe_idx_1 = min(self.step_counter + 1, total_len_traj - 1)
        safe_idx_5 = min(self.step_counter + 5, total_len_traj - 1)
        safe_idx_10 = min(self.step_counter + 10, total_len_traj - 1)
        safe_idx_20 = min(self.step_counter + 20, total_len_traj - 1)
        ref_pos1 = self.trajectory[safe_idx_1, :]
        ref_pos5 = self.trajectory[safe_idx_5, :]
        ref_pos10 = self.trajectory[safe_idx_10, :]
        ref_pos20 = self.trajectory[safe_idx_20, :]
        obs_parts = [
            ref_pos1 - state[:3],
            ref_pos5 - state[:3],
            ref_pos10 - state[:3],
            ref_pos20 - state[:3],
        ]

        obs_parts.extend([state[3:7], state[10:13], state[13:16]])
        for obs in static_obs:
            obs_parts.append(state[:3] - obs['pos'])
        ret = np.hstack(obs_parts).reshape(1, -1).astype("float32")

        return ret
    ################################################################################

    def step(self, action):
        self.current_action = action
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[:3]
        ref_pos = self.trajectory[self.step_counter, :]
        dist_error = np.linalg.norm(drone_pos - ref_pos)
        text_pos = [float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2]) + 0.1]
        if self.GUI and not self.visualised:
            self.visualised = True
            self.text_id = dummy_text("Track error: 0", text_pos, self.CLIENT)
            #self.draw_smooth_trajectory_curve(trajectory=self.trajectory)
            if self.obstacles:
                self._addObstacles()
        elif self.GUI:
            self.text_id = refreshed_text(f"Track error: {dist_error:.3f}", text_pos, self.CLIENT, self.text_id)
        return super().step(action)
    ################################################################################

    def _computeReward(self):
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[:3]
        ref_pos = self.trajectory[self.step_counter, :]
        latest_rl_action = self.action_buffer[-1][0, :]
        latest_safe_action = self.safe_action_buffer[-1][0, :]

        # ==========================================
        # ==========================================
        dist_error = np.linalg.norm(drone_pos - ref_pos)
        k_p = 1.5
        r_track = np.exp(- k_p * dist_error)  # 1 - dist_error
        self.r_track = r_track

        u_error = latest_safe_action-latest_rl_action
        delta_T = u_error[0]
        delta_xy = u_error[1:3]
        delta_z = u_error[3]

        k_i = 0.2

        penalty_T = 0.3 * delta_T ** 2
        penalty_xy = 1.0 * np.sum(delta_xy ** 2)
        penalty_z = 0.5 * delta_z ** 2

        self.raw_intervention = -(penalty_T + penalty_xy + penalty_z)
        self.r_safe = k_i * self.raw_intervention

        # ===============================
        # ===============================
        self.safe_u_T_abs = abs(latest_safe_action[0])
        self.safe_tau_x_abs = abs(latest_safe_action[1])
        self.safe_tau_y_abs = abs(latest_safe_action[2])
        self.safe_tau_z_abs = abs(latest_safe_action[3])

        self.rl_u_T_abs = abs(latest_rl_action[0])
        self.rl_tau_x_abs = abs(latest_rl_action[1])
        self.rl_tau_y_abs = abs(latest_rl_action[2])
        self.rl_tau_z_abs = abs(latest_rl_action[3])

        sat_tol = 0.95
        self.safe_action_saturation = float(np.max(np.abs(latest_safe_action)) > sat_tol)
        self.rl_action_saturation = float(np.max(np.abs(latest_rl_action)) > sat_tol)


        r = self.r_track + self.r_safe
        #r = self.r_track
        return r
    ################################################################################

    def _preprocessAction(self, action):
        rpm = np.zeros((self.NUM_DRONES, 4))  # (1,4)
        target = action[0, :]

        if self.ACT_TYPE == ActionType.TORQUE:
            THRUST = self.GRAVITY * (1 + 0.3 * target[0])
            tau_x = target[1] * self.MAX_XY_TORQUE
            tau_y = target[2] * self.MAX_XY_TORQUE
            tau_z = target[3] * self.MAX_Z_TORQUE
            return np.array([THRUST, tau_x, tau_y, tau_z])

        if self.ACT_TYPE == ActionType.RPM:
            rpm[0, :] = np.array(self.HOVER_RPM * (1 + 0.10 * target))
            return rpm
        if self.ACT_TYPE == ActionType.ATTITUDE_PID:
            state = self._getDroneStateVector(0)
            target_thrust = float(target[0])
            #target_thrust = np.array(self.HOVER_RPM * (1+0.05*target_thrust))
            target_rpy_rates = target[1:4]  # (3,)
            res, _, _ = self.ctrl[0].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3],
                target_rpy_rates=target_rpy_rates,
                target_thrust=target_thrust
            )
            rpm[0, :] = res  # (1,4)
            return rpm  # (4,)
    ################################################################################

    ################################################################################

    def _computeTerminated(self):
        if self.GUI:
            return False

        elif self.step_counter > self.discr_level - 2:
            self.reset_vars()
            return True
            # if self.eval_mode:
            #     return True
            # else:
            #     self.reset_vars()
            #     return True
        else:
            return False
    ################################################################################

    def _computeTruncated(self):
        drone_state = self._getDroneStateVector(0)
        drone_pos = drone_state[:3]
        ref_pos = self.trajectory[self.step_counter, :]
        dist_error = np.linalg.norm(drone_pos - ref_pos)
        # if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
        #     self.reset_vars()
        #     return True
        if dist_error > 5.0:
            self.reset_vars()
            return True
        # if not self.is_qp_solution:
        #     self.reset_vars()
        #     return True

        return False
    ################################################################################

    def _computeInfo(self):
        info = {
            "RL step_counter": self.step_counter,

            # reward
            "Reward of tracking": self.r_track,
            "Penalty for safe intervention": self.r_safe,
            "Raw safe intervention": self.raw_intervention,

            "Safe action abs/u_T": self.safe_u_T_abs,
            "Safe action abs/tau_x": self.safe_tau_x_abs,
            "Safe action abs/tau_y": self.safe_tau_y_abs,
            "Safe action abs/tau_z": self.safe_tau_z_abs,

            "RL action abs/u_T": self.rl_u_T_abs,
            "RL action abs/tau_x": self.rl_tau_x_abs,
            "RL action abs/tau_y": self.rl_tau_y_abs,
            "RL action abs/tau_z": self.rl_tau_z_abs,

            "Safe action saturation": self.safe_action_saturation,
            "RL action saturation": self.rl_action_saturation,
        }
        return info

    ################################################################################
    def draw_smooth_trajectory_curve(self, trajectory):
        #
        # if len(trajectory) > 1200:
        #     indices = np.linspace(0, len(trajectory) - 1, 1200, dtype=int)
        #     trajectory = trajectory[indices]

        for i in range(len(trajectory) - 1):
            p.addUserDebugLine(
                lineFromXYZ=trajectory[i],
                lineToXYZ=trajectory[i + 1],
                lineColorRGB=[0.95, 0.1, 0.1],
                lineWidth=2,
                lifeTime=0,
                physicsClientId=self.CLIENT
            )

    ################################################################################

    def _get_obstacle_pos(self, num_obstacles=3):
        if num_obstacles <= 0:
            return [], None

        len_traj = self.trajectory.shape[0]

        indices = np.linspace(
            0.1 * (len_traj - 1),
            0.9 * (len_traj - 1),
            num=num_obstacles,
            dtype=int
        )

        offsets = np.array([
            [0.01, 0.02, -0.02],
            [-0.02, 0.01, 0.03],
            [0.02, -0.02, -0.05]
        ])

        static_obs = []

        for i, indx_i in enumerate(indices):
            obs_base = self.trajectory[indx_i, :]

            offset = offsets[i % len(offsets)]
            obs_pos = obs_base + offset

            static_obs.append({
                'pos': obs_pos,
                'radius': 0.1
            })

        return static_obs, None

