#pragma once

#include <ceres/ceres.h>
#include <ros/assert.h>
#include <Eigen/Dense>
#include "../parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"

/**
 * @brief 重投影误差因子：关联两帧IMU位姿、相机IMU外参、特征点逆深度
 *        pose_i、pose_j、T_bc   7维
 *        特征点逆深度    1维
 */
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> {
 public:
  ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void check(double **parameters);

  Eigen::Vector3d pts_i, pts_j;         // 特征点在相机归一化平面上的坐标
  Eigen::Matrix<double, 2, 3> tangent_base;
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;
};
