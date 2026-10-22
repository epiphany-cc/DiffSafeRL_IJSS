#pragma once
#include "lib/traj_min_jerk.hpp"
#include "lib/traj_min_snap.hpp"
#include "traj_gen.hpp"
#include <Eigen/Eigen>

using namespace Eigen;

namespace traj {

    class TrajectoryWrapper {
        private: 
            traj::OptType opt_type;
            min_jerk::Trajectory min_jerk_trajectory;
            min_snap::Trajectory min_snap_trajectory;
        public:
            TrajectoryWrapper(traj::OptType opt_type, min_jerk::Trajectory min_jerk_trajectory, min_snap::Trajectory min_snap_trajectory);
            double getTotalDuration();
            Eigen::Vector3d getPos(double t);
            Eigen::Vector3d getVel(double t);
            Eigen::Vector3d getAcc(double t);
            Eigen::Vector3d getJuncPos(int juncIdx);
            Eigen::Vector3d getJuncVel(int juncIdx);
            Eigen::Vector3d getJuncAcc(int juncIdx);

    };
}