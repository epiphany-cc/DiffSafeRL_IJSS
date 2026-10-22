
#include "traj_gen.hpp"
#include "random_route_generator.hpp"


void traj::gen_trajectory_jerk(MatrixXd target_route, VectorXd target_timestamps, min_jerk::Trajectory & min_jerk_trajectory) 
{
    // states
    Matrix3d initialState, finalState;

    // Set initial position and final position, and set rest to zero
    initialState.setZero();
    finalState.setZero();
    initialState.col(0) << target_route.leftCols<1>();
    finalState.col(0) << target_route.rightCols<1>();

    // optimization class
    min_jerk::JerkOpt jerkOpt;

    // generate trajectory
    auto pointsBetween = target_route.block(0, 1, 3, target_route.cols() - 2);
    jerkOpt.reset(initialState, finalState, target_route.cols() - 1);
    jerkOpt.generate(pointsBetween, target_timestamps);
    jerkOpt.getTraj(min_jerk_trajectory);
}

void traj::gen_trajectory_snap(MatrixXd target_route, VectorXd target_timestamps, min_snap::Trajectory & min_snap_trajectory) {
    // states
    Eigen::Matrix<double, 3, 4> initialStateSnap, finalStateSnap;

    // Set initial position and final position, and set rest to zero
    initialStateSnap.setZero();
    finalStateSnap.setZero();
    initialStateSnap.col(0) << target_route.leftCols<1>();
    finalStateSnap.col(0) << target_route.rightCols<1>();

    // optimization class
    min_snap::SnapOpt snap_opt;

    // generate trajectory
    auto pointsBetween = target_route.block(0, 1, 3, target_route.cols() - 2);
    snap_opt.reset(initialStateSnap, finalStateSnap, target_route.cols() - 1);
    snap_opt.generate(pointsBetween, target_timestamps);
    snap_opt.getTraj(min_snap_trajectory);
}

void traj::gen_trajectory(MatrixXd target_route, VectorXd target_timestamps, OptType opt_type, min_jerk::Trajectory & min_jerk_trajectory , min_snap::Trajectory & min_snap_trajectory) {
    switch (opt_type) {
        case traj::OptType::JERK:
            traj::gen_trajectory_jerk(target_route, target_timestamps, min_jerk_trajectory);
            break;
        case traj::OptType::SNAP:
            traj::gen_trajectory_snap(target_route, target_timestamps, min_snap_trajectory);
            break;
        default: 
            return;
    }
}

VectorXd traj::allocateTime(const MatrixXd &wayPs,
                      double vel,
                      double acc)
{
    int N = (int)(wayPs.cols()) - 1;
    VectorXd durations(N);
    if (N > 0)
    {

        Eigen::Vector3d p0, p1;
        double dtxyz, D, acct, accd, dcct, dccd, t1, t2, t3;
        for (int k = 0; k < N; k++)
        {
            p0 = wayPs.col(k);
            p1 = wayPs.col(k + 1);
            D = (p1 - p0).norm();

            acct = vel / acc;
            accd = (acc * acct * acct / 2);
            dcct = vel / acc;
            dccd = acc * dcct * dcct / 2;

            if (D < accd + dccd)
            {
                t1 = sqrt(acc * D) / acc;
                t2 = (acc * t1) / acc;
                dtxyz = t1 + t2;
            }
            else
            {
                t1 = acct;
                t2 = (D - accd - dccd) / vel;
                t3 = dcct;
                dtxyz = t1 + t2 + t3;
            }

            durations(k) = dtxyz;
        }
    }

    return durations;
}
