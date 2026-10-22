import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from quadprog import solve_qp
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])

        self.SMC_LAMBDA = np.array([10.0, 10.0, 5.0])
        self.SMC_K = np.array([30000., 30000., 20000.])
        self.SMC_PHI = 0.5
        ### ------------------------------------------------------------------------
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])

        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                       control_timestep,
                       obs_pos,
                       obs_radius,
                       alpha0,
                       alpha1,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       target_thrust: float=None,
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        #####################################################################################################

        # target_thrust = max(0.0, float(target_thrust))
        # base_pwm = (np.sqrt(target_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        # rpm = self._dslPIDAttitudeRateControl(base_pwm=base_pwm,
        #                                       target_rpy_rates=target_rpy_rates,
        #                                       cur_rpy_rates=cur_ang_vel
        #                                       )
        #####################################################################################################
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         obs_pos,
                                                                         obs_radius,
                                                                         alpha0,
                                                                         alpha1,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        #####################################################################################################
        # rpm = self._dslSMCAttitudeControl(
        #     control_timestep,
        #     thrust,
        #     cur_quat,
        #     computed_target_rpy,
        #     target_rpy_rates
        # )
        #####################################################################################################
        # target_thrust, target_rpy_rates, target_euler = self.compute_thrust_and_target_rates(
        #                                                             a_safe=u_safe,
        #                                                             cur_quat=cur_quat,
        #                                                             curr_rpy=cur_rpy,
        #                                                             yaw_d=0,
        #                                                             )
        #####################################################################################################
        if target_thrust is None:
            rpm = self._dslPIDAttitudeControl(control_timestep=control_timestep,
                                              base_pwm=thrust,
                                              cur_quat=cur_quat,
                                              target_euler=computed_target_rpy,
                                              target_rpy_rates=target_rpy_rates
                                              )
        else:
            ## attitude control
            # base_pwm = (np.sqrt(target_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
            target_thrust = (target_thrust - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
            rpm = self._dslPIDAttitudeControl(
                control_timestep=control_timestep,
                base_pwm=target_thrust,
                cur_quat=cur_quat,
                target_euler=np.array(cur_rpy),
                target_rpy_rates=target_rpy_rates
            )
        #return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
        return rpm, None, None
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               obs_pos,
                               obs_radius,
                               alpha0,
                               alpha1,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        # target_thrust = self.get_safe_action(u_rl=target_thrust, obs_pos=obs_pos, cur_pos=cur_pos,
        #                                      cur_vel=cur_vel, obs_radius=obs_radius, alpha0=alpha0, alpha1=alpha1)
        ############################################################
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               base_pwm,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        #target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        #w, x, y, z = target_quat
        #target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        target_rotation = Rotation.from_euler('XYZ', target_euler, degrees=False).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = base_pwm + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    ################################################################################
    def _dslPIDAttitudeRateControl(self, base_pwm, target_rpy_rates, cur_rpy_rates):
        ang_vel_error = target_rpy_rates - cur_rpy_rates

        target_torques = np.multiply(self.D_COEFF_TOR, ang_vel_error)
        target_torques = np.clip(target_torques, -3200, 3200)

        pwm = base_pwm + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    ################################################################################
    def _dslSMCAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()

        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(),
                                                                                    target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])

        cur_rpy_rates = (cur_rpy - self.last_rpy) / control_timestep
        rpy_rates_e = target_rpy_rates - cur_rpy_rates
        self.last_rpy = cur_rpy
        # s_k = error_rate + lambda * error_position
        s_k = rpy_rates_e + np.multiply(self.SMC_LAMBDA, rot_e)

        q = 10.0
        epsilon = 0.8
        phi = self.SMC_PHI

        # delta_s = s(k+1) - s(k)
        delta_s = - q * control_timestep * s_k - epsilon * control_timestep * np.tanh(s_k / phi)

        # target_torques = K * tanh(s / phi)
        target_torques = np.multiply(self.SMC_K, (s_k + delta_s))
        #target_torques = np.multiply(self.SMC_K, np.tanh(s_k / self.SMC_PHI))

        #target_torques = np.clip(target_torques, -32000, 32000)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    ################################################################################
    def compute_thrust_and_target_rates(self, a_safe, curr_rpy, cur_quat, yaw_d):
        Kp_att = self.P_COEFF_TOR / self.D_COEFF_TOR

        Fx = self.mass * a_safe[0]
        Fy = self.mass * a_safe[1]
        Fz = self.mass * (a_safe[2] + self.g)
        target_thrust = np.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)

        if target_thrust < 1e-5:
            return 0.0, np.zeros(3)

        F_vec = np.array([Fx, Fy, Fz])
        z_b_des = F_vec / np.linalg.norm(F_vec)

        x_c_des = np.array([np.cos(yaw_d), np.sin(yaw_d), 0])

        y_b_des = np.cross(z_b_des, x_c_des)
        y_b_des = y_b_des / np.linalg.norm(y_b_des)
        x_b_des = np.cross(y_b_des, z_b_des)

        target_rotation = np.column_stack((x_b_des, y_b_des, z_b_des))  # 3x3

        target_euler = Rotation.from_matrix(target_rotation).as_euler('XYZ', degrees=False)

        w, x, y, z = cur_quat
        cur_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])

        target_rates = Kp_att * rot_e
        #target_rates = (target_euler - curr_rpy) / control_timestep
        target_RPM = np.sqrt(target_thrust / (4*self.KF) )
        return target_RPM, target_rates, target_euler

    ################################################################################
    def get_safe_action(self, u_rl, obs_pos, cur_pos, cur_vel, obs_radius, alpha0, alpha1):
        real_pos = cur_pos
        real_vel = cur_vel
        err_pos = real_pos - obs_pos
        G_vec = np.array([0, 0, 9.8])
        h0 = np.dot(err_pos, err_pos) - obs_radius ** 2
        h0_dot = 2 * np.dot(err_pos, real_vel)
        uav_mass = 0.027
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        u_rl = np.array(u_rl).reshape(3, )
        a_rl = (u_rl / uav_mass) - G_vec

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        H_scalar = - 2 * (alpha0 + alpha1) * np.dot(err_pos, real_vel) - 2 * np.dot(real_vel,
                                                                                    real_vel) - alpha0 * alpha1 * h0
        H = np.array([H_scalar], dtype=np.float64)

        # ---------------------------------------------------------
        # min 1/2·a^T·P·a - q^T·a
        # s.t. G^T * a >= H
        # ---------------------------------------------------------

        G_raw = (2.0 * err_pos).reshape((3, 1))
        G = np.array(G_raw, dtype=np.float64)

        P = np.array(np.eye(3) + np.eye(3) * 1e-8, dtype=np.float64)

        Q = np.array(a_rl, dtype=np.float64)

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        try:
            sol = solve_qp(P, Q, G, H)
            a_safe = sol[0]

            u_safe = uav_mass * (a_safe + G_vec)

        except ValueError as e:
            u_safe = u_rl

        return u_safe.reshape(3, )

    ################################################################################
    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM == 2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
