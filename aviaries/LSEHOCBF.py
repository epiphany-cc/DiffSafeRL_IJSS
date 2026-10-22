import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class DiffSafeFilter:
    def __init__(self, drone_model, J, J_inv, beta=1.2, gamma1=3, gamma2=3, max_T=0, max_tau=0):
        self.DRONE_MODEL = drone_model
        self.mass = self.getURDFParameter('m')
        self.J = J
        self.invJ = J_inv
        self.g_acc = 9.8
        self.e3 = np.array([0.0, 0.0, 1.0])

        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.max_T = max_T
        self.max_tau = max_tau

        self.rho_lse = 10.0

    def sigma_and_derivatives(self, s, b):
        A = max((self.beta - 1.0) * (b ** 2), 1e-6)
        K = 2.0 * np.sqrt(self.beta) * b + 0.1
        ratio = K / A

        tanh_val = np.tanh(ratio * s)
        sech2_val = 1.0 - tanh_val ** 2

        sigma = -A * tanh_val
        sigma_dot = -K * sech2_val
        sigma_ddot = 2.0 * K * ratio * tanh_val * sech2_val
        return sigma, sigma_dot, sigma_ddot

    def get_closed_form_safe_action(self, u_rl, pos, vel, ang_v, R, static_obs=None):
        if static_obs is None: static_obs = []

        obstacles = []
        for obs in static_obs:
            obstacles.append((np.array(obs['pos']), np.zeros(3), np.zeros(3), obs['radius']))

        num_obs = len(obstacles)

        if num_obs == 0:
            return u_rl, True, {'h0': [0], 'h1': [0], 'h2': [0]}

        l = R @ self.e3
        ang_v_body = R.T @ ang_v
        omega_skew = skew(ang_v_body)
        l_dot = R @ omega_skew @ self.e3
        gyro_moment = np.cross(ang_v_body, self.J @ ang_v_body)

        h_list, h_dot_list = [], []
        A_comp_i_list, B_T_i_list, B_tau_i_list = [], [], []

        for p_o, v_o, v_dot_o, b in obstacles:
            r = pos - p_o
            r_dot = vel - v_o

            s = np.dot(r, l)
            s_dot = np.dot(r_dot, l) + np.dot(r, l_dot)

            rho_val, rho_prime, rho_prime_prime = self.sigma_and_derivatives(s, b)

            beta_b2 = self.beta * (b ** 2)
            h_i = np.dot(r, r) - beta_b2 - rho_val
            h_list.append(h_i)

            h_i_dot = 2.0 * np.dot(r, r_dot) - rho_prime * s_dot
            h_dot_list.append(h_i_dot)

            s_ddot_drift = (
                    -np.dot(self.g_acc * self.e3 + v_dot_o, l)
                    + 2.0 * np.dot(r_dot, l_dot)
                    + np.dot(r, R @ omega_skew @ omega_skew @ self.e3)
                    + np.dot(r, R @ skew(self.e3) @ self.invJ @ gyro_moment)
            )

            h_i_ddot_drift = (
                    2.0 * np.dot(r_dot, r_dot)
                    - 2.0 * np.dot(r, self.g_acc * self.e3 + v_dot_o)
                    - rho_prime_prime * (s_dot ** 2)
                    - rho_prime * s_ddot_drift
            )
            A_comp_i_list.append(h_i_ddot_drift)

            B_T_i = (2.0 * s - rho_prime) / self.mass
            B_T_i_list.append(B_T_i)

            B_tau_i = rho_prime * r.T @ R @ skew(self.e3) @ self.invJ
            B_tau_i_list.append(B_tau_i)

        h_arr = np.array(h_list)
        h_dot_arr = np.array(h_dot_list)

        exp_terms = np.exp(-self.rho_lse * h_arr)
        sum_exp = np.sum(exp_terms)
        w_i = exp_terms / sum_exp

        H_val = -(1.0 / self.rho_lse) * (np.log(sum_exp))
        H_dot_val = np.sum(w_i * h_dot_arr)

        A_comp = np.sum(w_i * np.array(A_comp_i_list))
        B_T_comp = np.sum(w_i * np.array(B_T_i_list))
        B_tau_comp = np.sum(w_i[:, np.newaxis] * np.array(B_tau_i_list), axis=0)

        var_h_dot = np.sum(w_i * (h_dot_arr ** 2)) - (H_dot_val ** 2)
        Phi_2 = -self.rho_lse * var_h_dot

        Gamma = (self.gamma1 + self.gamma2) * H_dot_val + (self.gamma1 * self.gamma2) * H_val

        # =======================================================
        # =======================================================
        u_rl_flat = np.array(u_rl).flatten()
        T_RL = u_rl_flat[0]
        tau_RL = u_rl_flat[1:4]

        psi_RL = A_comp + B_T_comp * T_RL + np.dot(B_tau_comp, tau_RL) + Phi_2 + Gamma

        violation = max(0.0, -psi_RL)
        B_norm_sq = B_T_comp ** 2 + np.sum(B_tau_comp ** 2) + 1e-8

        T_star = T_RL + violation * B_T_comp / B_norm_sq
        tau_star = tau_RL + violation * B_tau_comp / B_norm_sq

        T_star = np.clip(T_star, 0.0, self.max_T)
        tau_star = np.clip(tau_star, -self.max_tau, self.max_tau)

        u_opt = np.hstack(([T_star], tau_star))

        u_opt_return = u_opt.reshape(1, 4) if isinstance(u_rl, (list, np.ndarray)) and len(
            np.shape(u_rl)) > 1 else u_opt

        cbf_info = {
            'h0': [H_val],
            'h1': [H_dot_val + self.gamma1 * H_val],
            'h2': [psi_RL + violation]
        }

        return u_opt_return, True, cbf_info

    def getURDFParameter(self,
                          parameter_name: str
                          ):
        """Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        #### Get the XML tree of the drone model to control ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+URDF)
        URDF_TREE = etxml.parse(path).getroot()
        #### Find and return the desired parameter #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name == 'inertia_matrix':
            ixx = float(URDF_TREE[1][0][2].attrib['ixx'])
            iyy = float(URDF_TREE[1][0][2].attrib['iyy'])
            izz = float(URDF_TREE[1][0][2].attrib['izz'])
            inertia_matrix = np.array([
                [ixx, 0, 0],
                [0, iyy, 0],
                [0, 0, izz]
            ])
            return inertia_matrix
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]