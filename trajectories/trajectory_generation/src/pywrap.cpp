#include <Eigen/Dense>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include "traj_gen.hpp"
#include "trajectory_wrapper.hpp"
#define LOG false
namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

/**
 * Calculates target trajectory.
 * 
 * Calculates given target waypoints as a 2-dimensional numpy array and target timestamps as a 1-dimensional numpy array. Assumes that values are of type double.
 * From: https://stackoverflow.com/questions/70012280/how-to-pass-a-vector-by-reference-in-pybind11-c
 **/ 
int calc_trajectory(
    py::array_t<double>& target_waypoints, 
    py::array_t<double>& target_timestamps,
    py::array_t<double>& result_waypoints,
    py::array_t<double>& result_timestamps) {
    
    if (LOG) {
        std::cout << "Checking for dimension correctness.." << std::endl;
    }
    // Check for input correctness ########################################
    if (target_waypoints.ndim() != 2)
        throw std::runtime_error("Target waypoints should be a 2-D Numpy array");
    if (result_waypoints.ndim() != 2)
        throw std::runtime_error("Result waypoints should be a 2-D Numpy array");
    if (target_timestamps.ndim() != 1)
        throw std::runtime_error("Target timestamps should be a 1-D Numpy array");
    if (result_timestamps.ndim() != 1)
        throw std::runtime_error("Result timestamps should be a 1-D Numpy array");
    
    auto waypoint_size = target_waypoints.shape()[0];
    if (waypoint_size != 3) 
        throw std::runtime_error("target_waypoints must be of shape 3 x N where N is the number of points and 3 is three-dimensional space.");
    if (result_waypoints.shape()[0] != waypoint_size) 
        throw std::runtime_error("trajectory_waypoints must be of shape 3 x N where N is the number of points and 3 is three-dimensional space.");

    auto n_target_points = target_waypoints.shape()[1];
    if (target_timestamps.shape()[0] + 1 != n_target_points) 
        throw std::runtime_error("Number of target waypoints should match number of target durations + 1.");

    auto n_result_points = result_waypoints.shape()[1];
    if (result_timestamps.shape()[0] != n_result_points) 
        throw std::runtime_error("Number of result waypoints should match number of result timestamps.");
    // ####################################################################

    if (LOG) {
        std::cout << "Pulling data.." << std::endl;
    }
    // PULL DATA ##########################################################
    double* target_waypoints_buf = (double*) target_waypoints.request().ptr;
    double* target_timestamps_buf = (double*) target_timestamps.request().ptr;
    double* result_waypoints_buf = (double*) result_waypoints.request().ptr;
    double* result_timestamps_buf = (double*) result_timestamps.request().ptr;
    // ####################################################################

    if (LOG) {
        std::cout << "Optimize trajectory.." << std::endl;
    }
    // OPTIMIZE_TRAJECTORY ################################################

    // config ++++++
    traj::OptType opt_type = traj::OptType::SNAP;
    // +++++++++++++

    // setup +++++++
    Eigen::MatrixXd target_waypoints_mat = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(target_waypoints_buf, 3, n_target_points);
    Eigen::VectorXd target_timestamps_vec = Eigen::Map<Eigen::VectorXd>(target_timestamps_buf, n_target_points - 1);

    min_jerk::Trajectory min_jerk_trajectory;
    min_snap::Trajectory min_snap_trajectory;
    // +++++++++++++

    if (LOG) {
        std::cout << "Gen trajectory.." << std::endl;
    }
    // gen +++++++++
    traj::gen_trajectory(
        target_waypoints_mat,
        target_timestamps_vec,
        opt_type,
        min_jerk_trajectory,
        min_snap_trajectory
    );
    // +++++++++++++

    // ####################################################################

    if (LOG) {
        std::cout << "Discretise output trajectory.." << std::endl;
    }
    // DISCRETIZE RESULT ##################################################
    traj::TrajectoryWrapper min_traj(opt_type, min_jerk_trajectory, min_snap_trajectory);
    auto total_duration = min_traj.getTotalDuration();
    for (double i = 0; i < (double) n_result_points; i++) 
    {
        // get position and timestamp.
        double time = (i / (n_result_points-1)) * total_duration;
        auto pos = min_traj.getPos(time);
        
        // save resulting waypoint
        int idx = (int) i;
        result_waypoints_buf[idx] = pos.x();
        result_waypoints_buf[idx + int(n_result_points)] = pos.y();
        result_waypoints_buf[idx + 2 * int(n_result_points)] = pos.z();

        // save resulting timestamp
        int index_timestamps = (int) i;
        result_timestamps_buf[index_timestamps] = time;
    }
    // ####################################################################
    return 0;
}

PYBIND11_MODULE(trajectory_cpp, m) {
    // optional module docstring
    m.doc() = "pybind11 calculator plugin";

    m.def("calc_trajectory", &calc_trajectory, "Calculates a trajectory for given target waypoints and timestamps.");
}