#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <algorithm>
#include <list>
#include <numeric>
#include <vector>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/assert.h>
#include <ros/console.h>

#include "parameters.h"

/**
 * @brief 特征点在某一帧的观测
 */
class FeaturePerFrame {
 public:
  FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td) {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5);
    velocity.y() = _point(6);
    cur_td = td;
  }
  double cur_td;        // IMU和相机之间的时间同步差
  Vector3d point;       // 归一化平面的坐标
  Vector2d uv;          // 像素坐标
  Vector2d velocity;    // 像素运动速度
  double z;             //
  bool is_used;         //
  double parallax;      // 视察
  MatrixXd A;           //
  VectorXd b;           //
  double dep_gradient;  //
};


/**
 * @brief 特征点在滑窗中的所有观测 
 */
class FeaturePerId {
 public:
  const int feature_id;                       // ID
  int start_frame;                            // 特征点在滑窗中的第一次观测
  vector<FeaturePerFrame> feature_per_frame;  // 特征点在滑窗中的所有观测

  int used_num;  // 特征被更踪的次数
  bool is_outlier;
  bool is_margin;
  double estimated_depth;   // 在当前滑窗中第一次被观测时的深度
  int solve_flag;  // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  Vector3d gt_p;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame), used_num(0), estimated_depth(-1.0), solve_flag(0) {
  }

  int endFrame();   // 获得 当前特征点在滑窗内的最后一帧观测的索引
};


/**
 * @brief 特征管理器
 */
class FeatureManager {
 public:
  FeatureManager(Matrix3d _Rs[]);

  void setRic(Matrix3d _ric[]);

  void clearState();

  int getFeatureCount();

  bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
  void debugShow();
  vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

  //void updateDepth(const VectorXd &x);
  void setDepth(const VectorXd &x);
  void removeFailures();
  void clearDepth(const VectorXd &x);
  VectorXd getDepthVector();
  void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier();

  list<FeaturePerId> feature;   // 特征列表
  int last_track_num;           // 

 private:
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
  const Matrix3d *Rs;         // 指向 estimator的 RS
  Matrix3d ric[NUM_OF_CAM];
};

#endif