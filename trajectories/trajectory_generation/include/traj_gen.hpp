#pragma once
#include "lib/traj_min_jerk.hpp"
#include "lib/traj_min_snap.hpp"
#include <Eigen/Eigen>


using namespace Eigen;

namespace traj {
    /**
     * Whether to use JERK or SNAP optimization
     **/
    enum OptType { JERK, SNAP };

    /**
     * Generates time stamps for given waypoints given velocity and acceleration.
     **/
    VectorXd allocateTime(const MatrixXd &wayPs,
                        double vel,
                        double acc);

    void gen_trajectory_jerk(MatrixXd target_route, VectorXd target_timestamps, min_jerk::Trajectory & min_jerk_trajectory);

    void gen_trajectory_snap(MatrixXd target_route, VectorXd target_timestamps, min_snap::Trajectory & min_snap_trajectory);

    void gen_trajectory(MatrixXd target_route, VectorXd target_timestamps, OptType opt_type, min_jerk::Trajectory & min_jerk_trajectory , min_snap::Trajectory & min_snap_trajectory);

    

}
