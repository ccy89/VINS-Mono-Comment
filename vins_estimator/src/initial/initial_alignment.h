#pragma once
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include "../factor/imu_factor.h"
#include "../feature_manager.h"
#include "../utility/utility.h"

using namespace Eigen;
using namespace std;

class ImageFrame {
 public:
  ImageFrame(){};
  ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, double _t) : t{_t}, is_key_frame{false} {
    points = _points;
  };
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points;  //
  double t;                                                         //
  Matrix3d R;                                                       // 图像帧时刻对应的IMU位姿  R_lb
  Vector3d T;                                                       // 图像帧的平移量  t_lc
  IntegrationBase *pre_integration;                                 //
  bool is_key_frame;                                                // 是否为关键帧
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x);