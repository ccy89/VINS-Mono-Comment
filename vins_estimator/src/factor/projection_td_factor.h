#pragma once

#include <ceres/ceres.h>
#include <ros/assert.h>
#include <Eigen/Dense>
#include "../parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"

/**
 * @brief 含有时间戳同步的重投影误差因子：关联两帧IMU位姿、相机IMU外参、特征点逆深度、相机和IMU时间差
 *        pose_i、pose_j、T_bc   7维
 *        特征点逆深度    1维
 *        相机和IMU时间差 1维
 */
class ProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1> {
 public:
  ProjectionTdFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
                     const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                     const double _td_i, const double _td_j, const double _row_i, const double _row_j);

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void check(double **parameters);

  Eigen::Vector3d pts_i, pts_j;             // 特征点在相机归一化平面上的坐标
  Eigen::Vector3d velocity_i, velocity_j;   // 特征点的像素速度
  double td_i, td_j;                        // 相机和IMU时间差
  Eigen::Matrix<double, 2, 3> tangent_base;
  double row_i, row_j;
  static Eigen::Matrix2d sqrt_info;         // 信息矩阵 LLT分解
  static double sum_t;
};
