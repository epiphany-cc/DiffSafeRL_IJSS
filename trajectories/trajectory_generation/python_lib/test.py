import numpy as np

from trajectory_cpp import calc_trajectory

N_RESULT_POINTS = 3

t_waypoints = np.array(
    [
        [0, 9.17827, 14.2934, 9.17827],
        [0, -7.98463, -15.3833, -7.98463],
        [0, 6.74148, -3.04313, 6.74148],
    ],
    dtype=np.float64,
)
t_durations = np.array([2, 2, 2], dtype=np.float64)
r_waypoints = np.zeros((3, N_RESULT_POINTS), dtype=np.float64)
r_timestamps = np.zeros(N_RESULT_POINTS, dtype=np.float64)


calc_trajectory(t_waypoints, t_durations, r_waypoints, r_timestamps)

print("Target waypoints: ", t_waypoints)
print("Resulting waypoints: ", r_waypoints)
