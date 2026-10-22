import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p
from trajectories import TrajectoryFactory, DiscretizedTrajectory
from scipy.interpolate import CubicSpline
from runnables.utils.get_eval_trajectories import Get_trajs
from quadprog import solve_qp

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

class CtrlAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 target_trajectory=None,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 120,
                 gui=False,
                 record=False,
                 obstacles=False,
                 log_positions=False,
                 episode_len_sec: int = 20,
                 discr_level: int = 1200,
                 eval_set: str = '',
                 trajectory_type: str = ''
                 ):
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        self.trajectory_type = trajectory_type
        self.get_eval_trajs = Get_trajs(base_folder=eval_set)
        enable_hocbfs = False
        enable_robust = False
        self.obstacles = obstacles
        self.EPISODE_LEN_SEC = episode_len_sec
        self.ctrl_freq = ctrl_freq
        self.NUM_DRONES = 1
        self.discr_level = discr_level
        # FOR DEVELOPMENT
        self.target_trajectory = target_trajectory
        # FOR EVAL
        self.GUI = gui
        self.trajectory = self.set_trajectory()
        self.current_action = None
        self.INIT_XYZS = self.trajectory[0:1, :]  # (1,3)
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
                        obstacles=obstacles,  # Add obstacles for RGB observations and/or FlyThruGate
                        user_debug_gui=False,  # Remove of RPM sliders from all single agent learning aviaries
                        log_positions=log_positions,
                        enable_hocbfs=enable_hocbfs,
                        enable_robust=enable_robust
                         )

        self.GUI = gui
        self.visualised = False
        self.discr_level = discr_level
        # # TRAJECTORY

        pos_x, pos_y, pos_z = self.INIT_XYZS[0]
        init_text_pos = [float(pos_x), float(pos_y), float(pos_z)]
        #self.text_id = dummy_text(f"Track error: {0:.3f}", init_text_pos, self.CLIENT)

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

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
            self.draw_smooth_trajectory_curve(trajectory=self.trajectory)
            if self.obstacles:
                self._addObstacles()
        elif self.GUI:
            self.text_id = refreshed_text(f"Track error: {dist_error:.3f}", text_pos, self.CLIENT, self.text_id)
        return super().step(action)

    ################################################################################

    def _preprocessAction(self, action):
        # rpm = action
        # THRUST = forces[0, 0] + forces[0, 1] + forces[0, 2] + forces[0, 3]
        #
        #
        # return np.array([THRUST, tau_x, tau_y, tau_z])
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])
    ################################################################################

    def draw_smooth_trajectory_curve(self, trajectory):

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

    def _get_obstacle_pos(self):

        len_traj = int(max(self.trajectory.shape[0], self.trajectory.shape[1]))
        indx_1 = int(0.2*len_traj)
        indx_2 = int(0.6*len_traj)
        indx_3 = int(0.8*len_traj)
        obs_1_base = self.trajectory[indx_1, :]
        obs_2_base = self.trajectory[indx_2, :]
        obs_3_base = self.trajectory[indx_3, :]
        obs_1 = obs_1_base + np.array([0.03, 0.05, 0.0])
        obs_2 = obs_2_base + np.array([-0.05, 0.04, 0.03])
        obs_3 = obs_3_base + np.array([0.03, -0.05, -0.05])
        static_obs = [
            {'pos': obs_1, 'radius': 0.2},
            {'pos': obs_2, 'radius': 0.2},
            {'pos': obs_3, 'radius': 0.2}
        ]

        dynamic_obs = [
            {'pos': [0.0, 0.0, 0.0], 'vel': [0.0, 0.0, 0.0], 'radius': 0.0}
        ]

        # dynamic_obs = [
        #     {'pos': [2.0, 0.0, 1.0], 'vel': [0.5, 0.0, 0.0], 'radius': 0.6},
        #     {'pos': [0.0, 3.0, 0.0], 'vel': [0.0, -0.2, 0.0], 'radius': 0.4}
        # ]

        return static_obs, None

    ################################################################################

    def set_trajectory(self):
        if self.GUI:
            n_discr = int(self.discr_level)
            take_off_discr = 0
            if take_off_discr != 0:
                trajectory, _ = self.get_eval_trajs.get_eval_trajs(n_discr=n_discr,
                                                                   ctrl_pyb=self.ctrl_freq,
                                                                   take_off_discr=take_off_discr,
                                                                   trajectory_type=self.trajectory_type,
                                                                   is_save_track_traj=False,
                                                                   is_save_plot_traj=False,
                                                                   )   # (n,3)
            else:
                trajectory = self.get_eval_trajs.get_eval_trajs(n_discr=n_discr,
                                                                ctrl_pyb=self.ctrl_freq,
                                                                take_off_discr=take_off_discr,
                                                                trajectory_type=self.trajectory_type,
                                                                is_save_track_traj=False,
                                                                is_save_plot_traj=False,
                                                                )   # (n,3)
        elif (not self.GUI) and (self.target_trajectory is not None):
            trajectory = self.target_trajectory

        else:
            ctrl_wps = TrajectoryFactory.get_random_trajectory(
                start=np.array([0, 0, 2]),
                n_discr_level=self.discr_level,
                n_ctrl_points=np.random.choice([15, 20]),
                std_dev_deg=np.random.choice([30, 50, 60]),
                distance_between_ctrl_points=1,
                init_dir=None,
                return_ctrl_points=False,
            )
            trajectory = np.array([x.coordinate for x in ctrl_wps])

        return trajectory

    ################################################################################

    def reset_vars(self):
        self.trajectory = self.set_trajectory()

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        if self.step_counter > self.discr_level - 2:
            self.reset_vars()
            return True
        else:
            return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        if self.step_counter > self.discr_level - 2:
            self.reset_vars()
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
