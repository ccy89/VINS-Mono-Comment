#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cstdlib>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;

/**
 * @brief 视觉惯性初始化的特征数据结构
 */
struct SFMFeature {
  bool state;                               // 特征点的状态（是否被三角化）
  int id;                                   // id
  vector<pair<int, Vector2d>> observation;  // 特征观测序列，结构为 <图像帧ID, 特征在图像的像素坐标>
  double position[3];                       // 3d坐标
  double depth;                             // 深度
};

/**
 * @brief 3D点的重投影误差
 */
struct ReprojectionError3D {
  ReprojectionError3D(double observed_u, double observed_v)
      : observed_u(observed_u), observed_v(observed_v) {}

  /**
   * @brief 重载 ceres 自动求导的 CostFuction 计算
   * 
   * @tparam T  变量数据类型
   * @param[in] camera_R    
   * @param[in] camera_T    
   * @param[in] point     
   * @param[in] residuals   
   */
  template <typename T>
  bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const {
    T p[3];
    ceres::QuaternionRotatePoint(camera_R, point, p);  // 四元数旋转3d点
    p[0] += camera_T[0];
    p[1] += camera_T[1];
    p[2] += camera_T[2];
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // 计算残差
    residuals[0] = xp - T(observed_u);
    residuals[1] = yp - T(observed_v);
    return true;
  }

  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y) {
    // 残差为2维，旋转四元数为4维，平移量为3维，3D点坐标为3维
    return (new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(
        new ReprojectionError3D(observed_x, observed_y)));
  }

  double observed_u;
  double observed_v;
};

class GlobalSFM {
 public:
  GlobalSFM();
  bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                 const Matrix3d relative_R, const Vector3d relative_T,
                 vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

 private:
  bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

  void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
  void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                            int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                            vector<SFMFeature> &sfm_f);

  int feature_num;
};