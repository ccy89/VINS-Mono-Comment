#pragma once

#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include "feature_manager.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "initial/initial_sfm.h"
#include "initial/solve_5pts.h"
#include "parameters.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/marginalization_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"

#include <opencv2/core/eigen.hpp>
#include <queue>
#include <unordered_map>

class Estimator {
 public:
  Estimator();

  void setParameter();

  // interface
  void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
  void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
  void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

  // internal
  void clearState();
  bool initialStructure();
  bool visualInitialAlign();
  bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
  void slideWindow();
  void solveOdometry();
  void slideWindowNew();
  void slideWindowOld();
  void optimization();
  void vector2double();
  void double2vector();
  bool failureDetection();

  enum SolverFlag {
    INITIAL,
    NON_LINEAR
  };

  enum MarginalizationFlag {
    MARGIN_OLD = 0,
    MARGIN_SECOND_NEW = 1
  };

  SolverFlag solver_flag;
  MarginalizationFlag marginalization_flag;
  Vector3d g;  // 重力 g_w
  MatrixXd Ap[2], backup_A;
  VectorXd bp[2], backup_b;

  // 相机和IMU外参
  Matrix3d ric[NUM_OF_CAM];
  Vector3d tic[NUM_OF_CAM];

  // 滑窗内的变量
  Vector3d Ps[(WINDOW_SIZE + 1)];
  Vector3d Vs[(WINDOW_SIZE + 1)];
  Matrix3d Rs[(WINDOW_SIZE + 1)];
  Vector3d Bas[(WINDOW_SIZE + 1)];
  Vector3d Bgs[(WINDOW_SIZE + 1)];
  double td;

  Matrix3d back_R0, last_R, last_R0;
  Vector3d back_P0, last_P, last_P0;
  std_msgs::Header Headers[(WINDOW_SIZE + 1)];

  IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];   // 滑窗内每一帧对应的IMU预积分量
  Vector3d acc_0, gyr_0;    // 最新传入的IMU数据

  vector<double> dt_buf[(WINDOW_SIZE + 1)];                     // 滑窗中相邻帧对应的dt
  vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];  // 滑窗中每一帧对应的加速度值序列
  vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];     // 滑窗中每一帧对应的角速度值序列

  int frame_count;
  int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

  FeatureManager f_manager;
  MotionEstimator m_estimator;
  InitialEXRotation initial_ex_rotation;

  bool first_imu;
  bool is_valid, is_key;
  bool failure_occur;

  vector<Vector3d> point_cloud;
  vector<Vector3d> margin_cloud;
  vector<Vector3d> key_poses;
  double initial_timestamp;

  // 用于ceres的优化变量
  double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
  double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
  double para_Feature[NUM_OF_F][SIZE_FEATURE];
  double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
  double para_Retrive_Pose[SIZE_POSE];
  double para_Td[1][1];
  double para_Tr[1][1];

  int loop_window_index;

  MarginalizationInfo *last_marginalization_info;          // 先验信息
  vector<double *> last_marginalization_parameter_blocks;  //

  map<double, ImageFrame> all_image_frame;
  IntegrationBase *tmp_pre_integration;

  //relocalization variable
  bool relocalization_info;
  double relo_frame_stamp;          // 闭环帧对应的关键帧的时间戳
  double relo_frame_index;          // 闭环帧在关键帧列表中的索引
  int relo_frame_local_index;       // 闭环帧对应的关键帧在滑窗中的索引
  vector<Vector3d> match_points;    // 前2维存储 闭环帧和对应关键帧的共视特征点在闭环帧的归一化坐标、第3维存储特征点ID
  double relo_Pose[SIZE_POSE];      // 闭环帧的位姿 (在前端VIO世界坐标系下)
  Matrix3d drift_correct_r;
  Vector3d drift_correct_t;

  // 滑窗优化前的 闭环帧和对应关键帧的相对位姿
  Vector3d prev_relo_t;
  Matrix3d prev_relo_r;

  // 滑窗优化后的 闭环帧和对应关键帧的相对位姿
  Vector3d relo_relative_t;
  Quaterniond relo_relative_q;
  double relo_relative_yaw;
};
