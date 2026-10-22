from .trajectory import Trajectory
from .square_linear_trajectory import SquareLinearTrajectory
from .polynomial_discretized_trajectory import PolynomialDiscretizedTrajectory
from .waypoint import Waypoint
from typing import List
import numpy as np
from .discretized_trajectory import (
    DiscretizedTrajectory,
    DiscreteTrajectoryFromContinuous,
    DiscretizedTrajectoryFromWaypoints,
    DiscretizedTrajFromNumpy,
)
from typing import Optional
from .random_trajectories import sample_random_ctrl_points
from typing import Tuple
import math

class WAYPOINT_BOOTH:
    T_WAYPOINTS_POLY = [
        Waypoint(coordinate=np.asarray([0, 0, 0.25]), timestamp=1),
        Waypoint(coordinate=np.asarray([0, 0, 0.25]), timestamp=2),
        Waypoint(coordinate=np.asarray([0, 0, 0.5]), timestamp=3),
        Waypoint(coordinate=np.asarray([0, 0, 0.25]), timestamp=4),
        Waypoint(coordinate=np.asarray([0, 0, 0]), timestamp=5),
    ]
    T_WAYPOINTS_POLY_ALL = [
        Waypoint(coordinate=np.asarray([0, 0, 0]), timestamp=0),
        Waypoint(coordinate=np.asarray([0, 0.5, 0.15]), timestamp=1),
        Waypoint(coordinate=np.asarray([0.25, 0.25, 0.075]), timestamp=2),
        Waypoint(coordinate=np.asarray([0, 0, 0]), timestamp=3),
    ]

class TrajectoryFactory:
    """
    Wrapper class for instantiating target trajectories.
    """
    @classmethod
    def get_linear_trajectory(
            cls,
            take_off_first_point=np.array([0, 0, 0]),
            take_off_last_point=np.array([0, 0, 0]),
            landing_first_point=np.array([0, 0, 0]),
            landing_last_point=np.array([0, 0, 0]),
            n_ctrl_points=8,
            n_discr_level=25,
            hover_count=5,
            take_off_only=True
    ) -> PolynomialDiscretizedTrajectory | Tuple[PolynomialDiscretizedTrajectory, List[Waypoint]]:
        t = np.linspace(0.0, 1.0, n_ctrl_points)
        take_off_ctrl_points = []
        landing_ctrl_points = []

        take_off_x = take_off_first_point[0]
        take_off_y = take_off_first_point[1]
        take_off_z = take_off_first_point[2] + t * (take_off_last_point[2] - take_off_first_point[2])
        for i in range(n_ctrl_points):
            take_off_coord = np.array([take_off_x, take_off_y, take_off_z[i]])
            take_off_ctrl_points.append(Waypoint(coordinate=take_off_coord, timestamp=i))
        for j in range(hover_count):
            take_off_ctrl_points.append(Waypoint(coordinate=take_off_last_point, timestamp=n_ctrl_points+j))

        landing_x = landing_first_point[0]
        landing_y = landing_first_point[1]
        landing_z = landing_first_point[2] + t * (landing_last_point[2] - landing_first_point[2])
        for j in range(hover_count):
            landing_ctrl_points.append(Waypoint(coordinate=landing_first_point, timestamp=j))
        for i in range(n_ctrl_points):
            landing_coord = np.array([landing_x, landing_y, landing_z[i]])
            landing_ctrl_points.append(Waypoint(coordinate=landing_coord, timestamp=i+hover_count))

        take_off_traj = cls.get_pol_discretized_trajectory(
            t_waypoints=take_off_ctrl_points,
            n_points_discretization_level=n_discr_level
        )
        take_off_traj_np = (take_off_traj[:].coordinate).T

        landing_traj = cls.get_pol_discretized_trajectory(
            t_waypoints=landing_ctrl_points,
            n_points_discretization_level=n_discr_level
        )
        landing_traj_np = (landing_traj[:].coordinate).T
        if take_off_only:
            return take_off_traj_np
        else:
            return take_off_traj_np, landing_traj_np

    @classmethod
    def get_circular_trajectory(
            cls,
            center: np.ndarray = np.array([0, 0, 5]),
            radius: float = 2.0,
            n_ctrl_points: int = 50,
            n_discr_level: int = 100,
            start_angle: float = 0.0,
            end_angle: float = 2 * math.pi,
            return_ctrl_points: bool = False
    ) -> PolynomialDiscretizedTrajectory | Tuple[PolynomialDiscretizedTrajectory, List[Waypoint]]:

        waypoints = []
        for i in range(n_ctrl_points):
            angle = start_angle + (end_angle - start_angle) * i / (n_ctrl_points - 1)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2] + 0.3*radius * math.sin(angle)
            waypoints.append(Waypoint(coordinate=np.array([x, y, z]), timestamp=i))

        traj = cls.get_pol_discretized_trajectory(
            t_waypoints=waypoints,
            n_points_discretization_level=n_discr_level
        )
        if return_ctrl_points:
            return traj, waypoints
        else:
            return traj

    @classmethod
    def get_helical_trajectory(
            cls,
            start: np.ndarray,
            radius: float = 2.0,
            height_per_turn: float = 1,
            num_turns: int = 3,
            points_per_turn: int = 15,
            n_discr_level: int = 100,
            return_ctrl_points: bool = False,
    ) -> PolynomialDiscretizedTrajectory:
        ctrl_points: List[Waypoint] = []
        total_points = num_turns * points_per_turn

        for i in range(int(total_points + 1)):
            angle = 2 * np.pi * (i / points_per_turn)
            x = start[0] + radius * np.cos(angle)
            y = start[1] + radius * np.sin(angle)
            z = start[2] + (i / total_points) * num_turns * height_per_turn
            timestamp = i
            ctrl_points.append(Waypoint(coordinate=np.array([x, y, z]), timestamp=timestamp))

        traj = cls.get_pol_discretized_trajectory(
            t_waypoints=ctrl_points,
            n_points_discretization_level=n_discr_level
        )

        if return_ctrl_points:
            return traj, ctrl_points
        else:
            return traj

    @classmethod
    def get_triangle_trajectory(
            cls,
            center: np.ndarray = np.array([0, 0, 3]),
            side_length: float = 4.0,
            n_ctrl_points_per_side: int = 10,
            n_discr_level: int = 90,
            return_ctrl_points: bool = False,
    ) -> PolynomialDiscretizedTrajectory | Tuple[PolynomialDiscretizedTrajectory, List[Waypoint]]:
        R = side_length / math.sqrt(3)
        angles = [0, 2 * math.pi / 3, 4 * math.pi / 3]
        vertices = []
        for angle in angles:
            x = center[0] + R * math.cos(angle)
            y = center[1] + R * math.sin(angle)
            vertices.append(np.array([x, y, center[2]]))

        ctrl_points: List[Waypoint] = []
        timestamp = 0

        for i in range(3):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % 3]

            for j in range(int(n_ctrl_points_per_side)):
                t = j / (int(n_ctrl_points_per_side) - 1)
                x = (1 - t) * start_vertex[0] + t * end_vertex[0]
                y = (1 - t) * start_vertex[1] + t * end_vertex[1]
                z = start_vertex[2]

                ctrl_points.append(Waypoint(
                    coordinate=np.array([x, y, z]),
                    timestamp=timestamp
                ))
                timestamp += 1

        traj = cls.get_pol_discretized_trajectory(
            t_waypoints=ctrl_points,
            n_points_discretization_level=n_discr_level
        )

        if return_ctrl_points:
            return traj, ctrl_points
        else:
            return traj

    @classmethod
    def get_pentagram_trajectory(
            cls,
            center: np.ndarray = np.array([0, 0, 2]),
            radius: float = 1.0,
            n_ctrl_points_per_side: int = 15,
            n_discr_level: int = 150,
            return_ctrl_points: bool = False,
    ) -> PolynomialDiscretizedTrajectory | Tuple[PolynomialDiscretizedTrajectory, List[Waypoint]]:
        vertices = []
        for i in range(5):
            angle = math.pi / 2 + i * (4 * math.pi / 5)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            vertices.append(np.array([x, y, center[2]]))

        ctrl_points = []
        timestamp = 0

        for idx in range(5):
            start_vertex = vertices[idx]
            end_vertex = vertices[(idx + 1) % 5]

            for j in range(int(n_ctrl_points_per_side)):
                t = j / (int(n_ctrl_points_per_side) - 1)
                x = (1 - t) * start_vertex[0] + t * end_vertex[0]
                y = (1 - t) * start_vertex[1] + t * end_vertex[1]
                z = start_vertex[2]


                ctrl_points.append(Waypoint(
                    coordinate=np.array([x, y, z]),
                    timestamp=timestamp
                ))
                timestamp += 1

        traj = cls.get_pol_discretized_trajectory(
            t_waypoints=ctrl_points,
            n_points_discretization_level=n_discr_level
        )

        if return_ctrl_points:
            return traj, ctrl_points
        else:
            return traj

    @classmethod
    def get_figure_eight_trajectory(
            cls,
            center: np.ndarray = np.array([0, 0, 5]),
            major_axis: float = 2.0,
            minor_axis: float = 1,
            n_ctrl_points: int = 50,
            n_discr_level: int = 100,
            num_loops: int = 1,
            phase_offset: float = math.pi/2,
            return_ctrl_points: bool = False
    ) -> PolynomialDiscretizedTrajectory | Tuple[PolynomialDiscretizedTrajectory, List[Waypoint]]:

        waypoints = []
        for idx in range(n_ctrl_points):
            theta = 2 * math.pi * num_loops * (idx / (n_ctrl_points - 1)) + phase_offset

            x = center[0] - major_axis * math.sin(theta)
            y = center[1] - minor_axis * math.sin(2 * theta)
            z = center[2] + 0.6 * math.sin(theta)

            waypoints.append(Waypoint(coordinate=np.array([x, y, z]), timestamp=idx))

        traj = cls.get_pol_discretized_trajectory(
            t_waypoints=waypoints,
            n_points_discretization_level=n_discr_level
        )

        if return_ctrl_points:
            return traj, waypoints
        else:
            return traj

    @classmethod
    def get_linear_square_trajectory(
        cls, square_scale: float = 1, time_scale: float = 1
    ) -> SquareLinearTrajectory:
        return SquareLinearTrajectory(square_scale, time_scale)

    @classmethod
    def get_linear_square_traj_discretized(
        cls,
        n_discretization_level: int = 100,
        square_scale: float = 1,
        time_scale: float = 1,
    ) -> DiscretizedTrajectory:
        sq_traj = SquareLinearTrajectory(square_scale, time_scale)
        discr_traj = DiscreteTrajectoryFromContinuous(sq_traj, n_discretization_level)
        return discr_traj

    @classmethod
    def get_pol_discretized_trajectory(
        cls,
        t_waypoints: Optional[List[Waypoint]] = None,
        n_points_discretization_level: Optional[int] = None,
    ) -> PolynomialDiscretizedTrajectory:

        if t_waypoints is None:
            t_waypoints = WAYPOINT_BOOTH.T_WAYPOINTS_POLY_ALL
        if n_points_discretization_level is None:
            n_points_discretization_level = 100

        return PolynomialDiscretizedTrajectory(
            t_waypoints, n_points_discretization_level
        )

    @classmethod
    def get_simple_smooth_trajectory(
        cls,
        starting_waypoint: Waypoint,
        t_waypoints: Optional[List[Waypoint]] = None,
        n_points_discretization_level: Optional[int] = None,
    ) -> DiscretizedTrajectory:
        if t_waypoints is None:
            t_waypoints = [starting_waypoint] + WAYPOINT_BOOTH.T_WAYPOINTS_POLY
        if n_points_discretization_level is None:
            n_points_discretization_level = 100

        return PolynomialDiscretizedTrajectory(
            t_waypoints, n_points_discretization_level
        )

    @classmethod
    def get_discr_from_wps(
        cls, t_waypoints: List[Waypoint]
    ) -> DiscretizedTrajectoryFromWaypoints:
        traj = DiscretizedTrajectoryFromWaypoints(t_waypoints)
        return traj

    @classmethod
    def get_discr_from_np(cls, fname: str) -> DiscretizedTrajectory:
        wps_np = np.load(fname)
        return DiscretizedTrajFromNumpy(wps_np=wps_np)

    @classmethod
    def waypoints_from_numpy(cls, t_waypoints: np.ndarray) -> List[Waypoint]:
        res = [
            Waypoint(t_waypoints[idx], timestamp=idx) for idx in range(len(t_waypoints))
        ]
        return res

    @classmethod
    def get_random_trajectory(
        cls,
        start: np.ndarray,
        std_dev_deg: float = 90,
        n_ctrl_points: int = 10,
        distance_between_ctrl_points: float = 1,
        n_discr_level: int = 100,
        init_dir: Optional[np.ndarray] = None,
        return_ctrl_points: bool = False,
    ) -> (
        PolynomialDiscretizedTrajectory
        | Tuple[PolynomialDiscretizedTrajectory, List[Waypoint]]
    ):

        # sample random feasible control points
        random_ctrl_points: List[Waypoint] = sample_random_ctrl_points(
            start_wp=start,
            std_dev_deg=std_dev_deg,
            n_ctrl_points=n_ctrl_points,
            distance_between_ctrl_point=distance_between_ctrl_points,
            init_dir=init_dir,
        )

        # compute polynomial minimum snap trajectory
        traj = cls.get_pol_discretized_trajectory(
            random_ctrl_points, n_points_discretization_level=n_discr_level
        )

        if return_ctrl_points:
            return traj, random_ctrl_points
        else:
            return traj
