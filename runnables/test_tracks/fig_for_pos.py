import numpy as np
import matplotlib.pyplot as plt
import os



def get_ax_range(ax1, imrl_pos, ref_pos):
    all_pos = np.vstack((imrl_pos,  ref_pos))
    mid_x = (all_pos[:, 0].max() + all_pos[:, 0].min()) * 0.5
    mid_y = (all_pos[:, 1].max() + all_pos[:, 1].min()) * 0.5
    mid_z = (all_pos[:, 2].max() + all_pos[:, 2].min()) * 0.5
    max_range = np.array([all_pos[:, 0].max() - all_pos[:, 0].min(),
                          all_pos[:, 1].max() - all_pos[:, 1].min(),
                          all_pos[:, 2].max() - all_pos[:, 2].min()]).max() / 2.0
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(0, mid_z + max_range)

def get_obstacle_pos(trajectory):

    len_traj = int(max(trajectory.shape[0], trajectory.shape[1]))
    indx_1 = int(0.2*len_traj)
    indx_2 = int(0.6*len_traj)
    indx_3 = int(0.8*len_traj)
    obs_1_base = trajectory[indx_1, :]
    obs_2_base = trajectory[indx_2, :]
    obs_3_base = trajectory[indx_3, :]
    obs_1 = obs_1_base + np.array([0.01, 0.02, -0.02])  #
    obs_2 = obs_2_base + np.array([-0.02, 0.01, 0.03])  #
    obs_3 = obs_3_base + np.array([0.02, -0.02, -0.05])  #

    static_obs = [
        {'pos': obs_1, 'radius': 0.1},
        {'pos': obs_2, 'radius': 0.1},
        {'pos': obs_3, 'radius': 0.1}
    ]

    return static_obs


def get_pos(trajectory_type: str = '', nth_traj: int = 0):
    ref_file_path = f"trajectories_for_eval/ref_{trajectory_type}/{nth_traj}.npy"
    DiffRL_file_path = f"trajectories_for_DiffRL/DiffRL_{trajectory_type}/{nth_traj}.npz"

    with np.load(DiffRL_file_path) as data:
        timestamps_data = (data['timestamps']).T  #  (N, 1)[cite: 1]
        DiffRL_states_data = (data['states'][0]).T  #  (N, 16)[cite: 1]
        DiffRL_cbf_value = (data['cbf_value'][0]).T


    time = timestamps_data.flatten()
    DiffRL_pos = DiffRL_states_data[:, 0:3]
    DiffRL_rpy = DiffRL_states_data[:, 3:6]
    ref_pos = np.load(ref_file_path)

    return DiffRL_pos, DiffRL_rpy, ref_pos, time, DiffRL_cbf_value

if __name__ == "__main__":
    is_save = False
    TRAJECTORY_TYPES = [
        "random", "circle", "helical",
        "eight", "triangle", 'pentagram'
    ]
    TRAJECTORY_TYPE = TRAJECTORY_TYPES[3]

    save_dir = f"eval_results/{TRAJECTORY_TYPE}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    DiffRL_pos, DiffRL_rpy, ref_pos, time, DiffRL_cbf_value = get_pos(TRAJECTORY_TYPE, 0)
    static_obs = get_obstacle_pos(trajectory=ref_pos)
    DiffRL_error = DiffRL_pos - ref_pos
    DiffRL_total_error = np.linalg.norm(DiffRL_error, axis=1)

    num_obs = len(static_obs)
    distances_DiffRL = np.zeros((DiffRL_pos.shape[0], num_obs))

    for i, obs in enumerate(static_obs):
        obs_pos = obs['pos']
        distances_DiffRL[:, i] = np.linalg.norm(DiffRL_pos - obs_pos, axis=1)
    DiffRL_min_distances = np.min(distances_DiffRL, axis=1)


    color_ref = 'black'
    color_imrl = '#d62728'  #
    color_pid = '#1f77b4'  #
    color_est = '#1f77b4'  #

    # ========================================================
    # ========================================================
    fig1 = plt.figure(figsize=(16, 9))
    ax1 = fig1.add_subplot(111, projection='3d')
    light_gray = (1, 1, 1, 1.0)
    ax1.xaxis.set_pane_color(light_gray)
    ax1.yaxis.set_pane_color(light_gray)
    ax1.zaxis.set_pane_color(light_gray)
    ax1.set_box_aspect([1.2, 1.2, 0.8])


    x0, y0, z0 = DiffRL_pos[0]
    ax1.plot([x0, x0], [y0, y0], [0, z0], color='gray', linestyle=':')
    ax1.scatter(x0, y0, 0, color='purple', marker='^', s=50, label='Takeoff position')
    ax1.scatter(x0, y0, z0, color='green', marker='o', s=40, label='Initial position')
    for i, obs in enumerate(static_obs):
        pos = obs['pos']  #
        radius = obs['radius']  #


        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)

        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.6, edgecolor='none')

    ax1.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], color=color_ref, linestyle='--', linewidth=2.5,
             label='Reference trajectory')
    ax1.plot(DiffRL_pos[:, 0], DiffRL_pos[:, 1], DiffRL_pos[:, 2], color=color_imrl, linestyle='-', linewidth=3,
             label='The proposed method')
    ax1.plot([x0, x0], [y0, y0], [0, z0], color=color_imrl, linestyle='-', linewidth=3)
    ax1.scatter([], [], [], color='blue', marker='.', s=300, label='Obstacles')
    #get_ax_range(ax1, imrl_pos, ref_pos)
    ax1.set_xlabel('x (m)', fontsize=12, labelpad=5)  #
    ax1.set_ylabel('y (m)', fontsize=12, labelpad=5)
    ax1.set_zlabel('z (m)', fontsize=12, labelpad=2)
    ax1.xaxis.set_tick_params(labelsize=12)  #
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.zaxis.set_tick_params(labelsize=12)
    ax1.view_init(elev=20, azim=45)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False,
               columnspacing=2,  #
               handletextpad=1,  #
               fontsize=12,
               labelspacing=0.8  #
               )
    if is_save:
        fig1.savefig(os.path.join(save_dir,  f'{TRAJECTORY_TYPE}_traj_3d.pdf'))

    # ========================================================
    fig2, ax2 = plt.subplots(figsize=(12, 3))  #
    #
    ax2.plot(time, DiffRL_total_error, color=color_imrl, linewidth=2.5, label='Tracking error')

    ax2.set_ylabel('Tracking Error (m)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.5)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, )
    plt.tight_layout()
    if is_save:
        fig2.savefig(os.path.join(save_dir,  f'{TRAJECTORY_TYPE}_tracking_error.pdf'), bbox_inches='tight')

    # ========================================================
    # fig 4
    # ========================================================
    fig4, ax4 = plt.subplots(figsize=(12, 3))

    ax4.plot(time, DiffRL_min_distances, color=color_imrl, linewidth=2.5, label='Distance to closest obstacle')

    radius = static_obs[0]['radius']
    #
    ax4.axhline(y=radius, color=color_est, linestyle='--', linewidth=2.5, label=f'Safe boundary (radius=0.1m)')
    #
    ax4.fill_between(time, 0, radius, color=color_est, alpha=0.4, label='Unsafe region')
    #
    ax4.set_xlabel('Time (s)', fontsize=14)
    ax4.set_ylabel('Distance (m)', fontsize=14)
    ax4.xaxis.set_tick_params(labelsize=12)
    ax4.yaxis.set_tick_params(labelsize=12)

    ax4.set_xlim(0, 20)
    ax4.set_ylim(0, )
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.legend(loc='upper right', fontsize=12, framealpha=0.5)
    plt.tight_layout()
    if is_save:
        fig4.savefig(os.path.join(save_dir, f'{TRAJECTORY_TYPE}_min_distance_to_obstacles.pdf'), bbox_inches='tight')

    # ========================================================
    # ========================================================
    fig5, ax5 = plt.subplots(figsize=(12, 3))
    imrl_h0_data = DiffRL_cbf_value[:, 0]

    ax5.plot(time, imrl_h0_data, color=color_imrl, linewidth=2.5, label='Composite HOCBF value')
    ax5.axhline(y=0, color=color_est, linestyle='--', linewidth=2.5, label='Safe boundary ($H(x)=0$)')
    ax5.set_xlabel('Time (s)', fontsize=12)
    ax5.xaxis.set_tick_params(labelsize=12)
    ax5.yaxis.set_tick_params(labelsize=12)
    ax5.grid(True, linestyle=':')
    ax5.legend(loc='upper right', fontsize=12, framealpha=0.5)
    ax5.set_xlim(0, 20)
    ax5.set_ylim(-0.5,)
    plt.tight_layout()
    if is_save:
        fig5.savefig(os.path.join(save_dir,  f'{TRAJECTORY_TYPE}_composite_hocbf_value.pdf'), bbox_inches='tight')


    # ========================================================
    # fig 7
    # ========================================================
    fig7, ax7 = plt.subplots(figsize=(12, 3))
    DiffRL_rpy[-20:, :] = DiffRL_rpy[-21, :]

    ax7.plot(time, DiffRL_rpy[:, 0], label='Roll', color='orange', linewidth=3)
    ax7.plot(time, DiffRL_rpy[:, 1], label='Pitch', color='purple', linewidth=3)
    ax7.plot(time, DiffRL_rpy[:, 2], label='Yaw', color='cyan', linewidth=3)

    ax7.set_xlabel('Time (s)', fontsize=12)
    ax7.set_ylabel('Attitude (rad)', fontsize=12)
    ax7.xaxis.set_tick_params(labelsize=12)
    ax7.yaxis.set_tick_params(labelsize=12)

    ax7.legend(loc='upper right', fontsize=12, framealpha=0.5)
    ax7.grid(True, linestyle=':')


    ax7.set_xlim(0, int(ref_pos.shape[0] / 120))

    plt.tight_layout()

    if is_save:
        fig7_path = os.path.join(save_dir, f'{TRAJECTORY_TYPE}_DiffRL_attitude.pdf')
        fig7.savefig(fig7_path, bbox_inches='tight', pad_inches=0.05)

    plt.show()

