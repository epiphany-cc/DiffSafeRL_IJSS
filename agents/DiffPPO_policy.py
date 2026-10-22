import torch
import torch.nn as nn
from functools import partial
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources


def quat_to_rot_mat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.zeros(B, 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R


class SafePPOPolicy(ActorCriticPolicy):

    nominal_action_dim = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.use_sde:
            raise NotImplementedError("SafePPOPolicy use_sde=False")

        env_action_dim = int(np.prod(self.action_space.shape))
        if env_action_dim != 8:
            raise ValueError(
                f"SafePPOPolicy 8 [u_safe, u_rl]"
                f"action_space {env_action_dim}"
            )

        # ==========================================================
        # ==========================================================
        M, L, J_mat, J_INV, KF, KM, T2W = self.parse_urdf_parameters(drone_model="cf2x")

        self.mass = M
        g_acc = 9.8
        self.GRAVITY = g_acc * self.mass

        MAX_RPM = np.sqrt((T2W * self.GRAVITY) / (4 * KF))
        MAX_THRUST = 4 * KF * MAX_RPM ** 2
        MAX_XY_TORQUE = (2 * L * KF * MAX_RPM ** 2) / np.sqrt(2)
        MAX_Z_TORQUE = 2 * KM * MAX_RPM ** 2

        self.max_T = MAX_THRUST
        max_tau_val = [MAX_XY_TORQUE, MAX_XY_TORQUE, MAX_Z_TORQUE]

        self.register_buffer("e3", torch.tensor([0.0, 0.0, 1.0]))
        self.register_buffer("gravity_vec", torch.tensor([0.0, 0.0, g_acc]))
        self.register_buffer("J", torch.tensor(J_mat, dtype=torch.float32))
        self.register_buffer("J_INV", torch.tensor(J_INV, dtype=torch.float32))
        self.register_buffer("max_tau", torch.tensor(max_tau_val, dtype=torch.float32))

        # ==========================================================
        # ==========================================================
        self.beta = 1.05
        self.gamma1 = 2.0
        self.gamma2 = 3.0
        self.rho_lse = 50.0
        self.obs_radius = 0.1
        self.num_obs = 3

        self.current_cbf_info = {
            "h0": [0.0],
            "h1": [0.0],
            "h2": [0.0],
        }

    def _build(self, lr_schedule):
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_dist = DiagGaussianDistribution(self.nominal_action_dim)
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi,
            log_std_init=self.log_std_init,
        )

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def sigma_and_derivatives_pt(self, s, b):
        A = max((self.beta - 1.0) * (b ** 2), 1e-6)
        K = 2.0 * np.sqrt(self.beta) * b + 0.1
        ratio = K / A

        tanh_val = torch.tanh(ratio * s)
        sech2_val = 1.0 - tanh_val ** 2

        sigma = -A * tanh_val
        sigma_dot = -K * sech2_val
        sigma_ddot = 2.0 * K * ratio * tanh_val * sech2_val
        return sigma, sigma_dot, sigma_ddot

    def parse_urdf_parameters(self, drone_model="cf2x"):
        urdf_file = drone_model + ".urdf"
        path = pkg_resources.resource_filename("gym_pybullet_drones", "assets/" + urdf_file)
        URDF_TREE = etxml.parse(path).getroot()

        M = float(URDF_TREE[1][0][1].attrib["value"])
        L = float(URDF_TREE[0].attrib["arm"])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib["thrust2weight"])

        IXX = float(URDF_TREE[1][0][2].attrib["ixx"])
        IYY = float(URDF_TREE[1][0][2].attrib["iyy"])
        IZZ = float(URDF_TREE[1][0][2].attrib["izz"])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib["kf"])
        KM = float(URDF_TREE[0].attrib["km"])

        return M, L, J, J_INV, KF, KM, THRUST2WEIGHT_RATIO

    def cbf_projection(self, obs: torch.Tensor, u_rl_norm: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        T_RL = self.GRAVITY * (1.0 + 0.3 * u_rl_norm[:, 0])
        tau_RL = u_rl_norm[:, 1:4] * self.max_tau

        quat = obs[:, 0, 12:16]
        vel = obs[:, 0,  16:19]
        ang_v_world = obs[:, 0, 19:22]

        R = quat_to_rot_mat(quat)
        l = R[:, :, 2]

        ang_v = torch.matmul(R.transpose(1, 2), ang_v_world.unsqueeze(-1)).squeeze(-1)

        e3_batch = self.e3.expand(B, -1)

        omega_cross_e3 = torch.cross(ang_v, e3_batch, dim=1)
        l_dot = torch.matmul(R, omega_cross_e3.unsqueeze(-1)).squeeze(-1)

        J_ang_v = torch.matmul(self.J, ang_v.unsqueeze(-1)).squeeze(-1)  # J\omega
        gyro_moment = torch.cross(ang_v, J_ang_v, dim=1)  # \omega \times J\omega
        invJ_gyro = torch.matmul(self.J_INV, gyro_moment.unsqueeze(-1)).squeeze(-1)  # J^{-1}(\omega \times J\omega)

        omega_cross_omega_cross_e3 = torch.cross(ang_v, omega_cross_e3, dim=1)
        term3_world = torch.matmul(R, omega_cross_omega_cross_e3.unsqueeze(-1)).squeeze(-1)

        h_list, h_dot_list, A_i_list, B_T_i_list, B_tau_i_list = [], [], [], [], []

        # ==========================================================
        # ==========================================================
        for i in range(self.num_obs):
            start_idx = 22 + i * 3
            r_i = obs[:, 0, start_idx: start_idx + 3]

            s_i = torch.sum(r_i * l, dim=1)
            s_dot_i = torch.sum(vel * l, dim=1) + torch.sum(r_i * l_dot, dim=1)

            rho_val, rho_prime, rho_prime_prime = self.sigma_and_derivatives_pt(s_i, self.obs_radius)

            beta_b2 = self.beta * (self.obs_radius ** 2)
            h_i = torch.sum(r_i * r_i, dim=1) - beta_b2 - rho_val
            h_i_dot = 2.0 * torch.sum(r_i * vel, dim=1) - rho_prime * s_dot_i

            # ----------------------------------------------------
            # ----------------------------------------------------
            term3 = torch.sum(r_i * term3_world, dim=1)  # r_i^T R(\omega \times (\omega \times e_3))

            r_body = torch.matmul(R.transpose(1, 2), r_i.unsqueeze(-1)).squeeze(-1)
            r_body_cross_e3 = torch.cross(r_body, e3_batch, dim=1)  # R^T r_i x e3

            # (R^T r_i \times e_3)^T J^{-1} (\omega \times J\omega)
            term4 = torch.sum(r_body_cross_e3 * invJ_gyro, dim=1)

            s_ddot_drift = (
                    -torch.sum(self.gravity_vec.expand(B, -1) * l, dim=1)
                    + 2.0 * torch.sum(vel * l_dot, dim=1)
                    + term3 + term4
            )

            h_i_ddot_drift = (
                    2.0 * torch.sum(vel * vel, dim=1)
                    - 2.0 * torch.sum(r_i * self.gravity_vec.expand(B, -1), dim=1)
                    - rho_prime_prime * (s_dot_i ** 2)
                    - rho_prime * s_ddot_drift
            )

            # ----------------------------------------------------
            # ----------------------------------------------------
            B_T_i = (2.0 * s_i - rho_prime) / self.mass
            B_tau_i = rho_prime.unsqueeze(1) * torch.matmul(r_body_cross_e3.unsqueeze(1), self.J_INV).squeeze(1)

            h_list.append(h_i)
            h_dot_list.append(h_i_dot)
            A_i_list.append(h_i_ddot_drift)
            B_T_i_list.append(B_T_i)
            B_tau_i_list.append(B_tau_i)

        h_arr = torch.stack(h_list, dim=1)
        h_dot_arr = torch.stack(h_dot_list, dim=1)
        A_arr = torch.stack(A_i_list, dim=1)
        B_T_arr = torch.stack(B_T_i_list, dim=1)
        B_tau_arr = torch.stack(B_tau_i_list, dim=1)

        # ==========================================================
        # ==========================================================
        max_h, _ = torch.max(-self.rho_lse * h_arr, dim=1, keepdim=True)
        exp_terms = torch.exp(-self.rho_lse * h_arr - max_h)
        sum_exp = torch.sum(exp_terms, dim=1, keepdim=True)

        w_i = exp_terms / sum_exp

        H_val = -(1.0 / self.rho_lse) * (torch.log(sum_exp) + max_h).squeeze(1)
        H_dot_val = torch.sum(w_i * h_dot_arr, dim=1)

        A_comp = torch.sum(w_i * A_arr, dim=1)
        B_T_comp = torch.sum(w_i * B_T_arr, dim=1)
        B_tau_comp = torch.sum(w_i.unsqueeze(-1) * B_tau_arr, dim=1)

        # ==========================================================
        # ==========================================================
        var_h_dot = torch.sum(w_i * (h_dot_arr ** 2), dim=1) - (H_dot_val ** 2)
        Phi_2 = -self.rho_lse * var_h_dot
        Gamma = (self.gamma1 + self.gamma2) * H_dot_val + (self.gamma1 * self.gamma2) * H_val

        # ==========================================================
        # ==========================================================
        tau_dot_B = torch.sum(B_tau_comp * tau_RL, dim=1)
        psi_RL = A_comp + B_T_comp * T_RL + tau_dot_B + Phi_2 + Gamma

        h0_val = H_val[0].item()
        h1_val = (self.gamma1 * H_val + H_dot_val)[0].item()
        h2_val = psi_RL[0].item()

        self.current_cbf_info = {
            'h0': [h0_val],
            'h1': [h1_val],
            'h2': [h2_val]
        }

        violation = torch.relu(-psi_RL)
        B_norm_sq = B_T_comp ** 2 + torch.sum(B_tau_comp ** 2, dim=1) + 1e-8

        T_star = T_RL + violation * B_T_comp / B_norm_sq
        tau_star = tau_RL + violation.unsqueeze(1) * B_tau_comp / B_norm_sq.unsqueeze(1)

        T_star = torch.clamp(T_star, 0.0, self.max_T)
        tau_star = torch.clamp(tau_star, -self.max_tau, self.max_tau)

        # ==========================================================
        # ==========================================================
        T_norm = (T_star / self.GRAVITY - 1.0) / 0.3
        tau_norm = tau_star / self.max_tau

        u_star_norm = torch.cat([T_norm.unsqueeze(1), tau_norm], dim=1)
        u_star_norm = torch.clamp(u_star_norm, -1.0, 1.0)

        return u_star_norm

    def _get_latents(self, obs: torch.Tensor):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def _compose_action(self, obs: torch.Tensor, u_rl_raw: torch.Tensor) -> torch.Tensor:
        u_rl_norm = torch.clamp(u_rl_raw, -1.0, 1.0)
        u_safe_norm = self.cbf_projection(obs, u_rl_norm)

        action_8d = torch.cat([u_safe_norm, u_rl_raw], dim=1)
        return action_8d.reshape((-1, *self.action_space.shape))

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        latent_pi, latent_vf = self._get_latents(obs)
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        u_rl_raw = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(u_rl_raw)

        actions = self._compose_action(obs, u_rl_raw)
        return actions, values, log_prob

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        latent_pi, latent_vf = self._get_latents(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)

        actions_flat = actions.reshape(actions.shape[0], -1)
        u_rl_raw = actions_flat[:, 4:8]

        log_prob = distribution.log_prob(u_rl_raw)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        return values, log_prob, entropy
