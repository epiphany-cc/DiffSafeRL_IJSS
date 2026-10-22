#include "trajectory_wrapper.hpp"

traj::TrajectoryWrapper::TrajectoryWrapper(traj::OptType opt_type, min_jerk::Trajectory min_jerk_trajectory, min_snap::Trajectory min_snap_trajectory) 
{
    this->opt_type = opt_type;
    this->min_jerk_trajectory = min_jerk_trajectory;
    this->min_snap_trajectory = min_snap_trajectory;
}

double traj::TrajectoryWrapper::getTotalDuration() 
{
    if (this->opt_type == traj::OptType::JERK)
    { 
        return this->min_jerk_trajectory.getTotalDuration();
    } 
    else 
    {
        return this->min_snap_trajectory.getTotalDuration();
    }
}

Eigen::Vector3d traj::TrajectoryWrapper::getPos(double t) 
{
    if (this->opt_type == traj::OptType::JERK)
    { 
        return this->min_jerk_trajectory.getPos(t);
    } 
    else 
    {
        return this->min_snap_trajectory.getPos(t);
    }
}

Eigen::Vector3d traj::TrajectoryWrapper::getVel(double t) {throw std::runtime_error("Not implemented");}
Eigen::Vector3d traj::TrajectoryWrapper::getAcc(double t) {throw std::runtime_error("Not implemented");}
Eigen::Vector3d traj::TrajectoryWrapper::getJuncPos(int juncIdx) {throw std::runtime_error("Not implemented");}
Eigen::Vector3d traj::TrajectoryWrapper::getJuncVel(int juncIdx) {throw std::runtime_error("Not implemented");}
Eigen::Vector3d traj::TrajectoryWrapper::getJuncAcc(int juncIdx) {throw std::runtime_error("Not implemented");}