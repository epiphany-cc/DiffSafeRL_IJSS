import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trajectories import TrajectoryFactory, DiscretizedTrajectory
from waypoint import Waypoint
def generate_circular_trajectory(
        radius: float = 1.0,
        center: tuple = (0.0, 0.0, 0.0),
        normal: tuple = (0.0, 0.0, 1.0),
        num_points: int = 100,
        start_angle: float = 0.0,
        end_angle: float = 2 * np.pi
) -> np.ndarray:
    normal = np.array(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    if np.abs(normal[0]) < np.abs(normal[1]):
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = np.cross(normal, vec)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    angles = np.linspace(start_angle, end_angle, num_points, endpoint=True)


    center = np.array(center, dtype=np.float64)
    trajectory = np.zeros((num_points, 3), dtype=np.float64)
    for i, theta in enumerate(angles):
        trajectory[i] = center + radius * (np.cos(theta) * u + np.sin(theta) * v)


    return trajectory


if __name__ == "__main__":
    traj_xy = generate_circular_trajectory(
        radius=2.0,
        center=(1.0, 2.0, 1.0),
        normal=(0, 0, 1),
        num_points=15
    )
    line1_end = traj_xy[0:1, :]
    line1_start = np.copy(line1_end)
    line1_start[:, 2] = 0
    line2_start = traj_xy[-1:, :]
    line2_end = np.copy(line2_start)
    line2_end[:, 2] = 0
    traj_xy = np.vstack([line1_start, line1_end, traj_xy, line2_start, line2_end])
    waypoints = []
    for i in range(traj_xy.shape[0]):
        waypoints.append(Waypoint(coordinate=traj_xy[i, :], timestamp=i))
    #waypoints_from_numpy = TrajectoryFactory.waypoints_from_numpy(traj_xy)


    traj_inclined = generate_circular_trajectory(
        radius=1.5,
        center=(0, 0, 0),
        normal=(1, 1, 0),
        num_points=150
    )

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(traj_xy[:, 0], traj_xy[:, 1], traj_xy[:, 2], 'b-', label='XY-plant circle')
    ax1.scatter(traj_xy[0, 0], traj_xy[0, 1], traj_xy[0, 2], c='r', s=50, label='start point')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('XY-plant trajectory')
    ax1.legend()
    ax1.grid(True)

    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.plot(traj_inclined[:, 0], traj_inclined[:, 1], traj_inclined[:, 2], 'g-', label='')
    # ax2.scatter(traj_inclined[0, 0], traj_inclined[0, 1], traj_inclined[0, 2], c='r', s=50, label='start point')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')
    # ax2.set_title('')
    # ax2.legend()
    # ax2.grid(True)

    plt.tight_layout()
    plt.show()