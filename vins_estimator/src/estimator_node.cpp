#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <stdio.h>
#include <condition_variable>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;              // IMU数据队列
queue<sensor_msgs::PointCloudConstPtr> feature_buf;   // 特征数据队列
queue<sensor_msgs::PointCloudConstPtr> relo_buf;      // 重定位帧特征数据
int sum_of_wait = 0;

// 线程互斥量
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;


/**
 * @brief IMU 中值积分
 * 
 * @param[in] imu_msg   IMU数据
 */
void predict(const sensor_msgs::ImuConstPtr &imu_msg) {
  double t = imu_msg->header.stamp.toSec();   // 获得 当前IMU的时间戳
  if (init_imu) {   
    // 如果是第一帧IMU数据
    latest_time = t;
    init_imu = 0;
    return;
  }
  double dt = t - latest_time;    // Δt

  // 获取加速度值
  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  Eigen::Vector3d linear_acceleration(dx, dy, dz);
  
  // 获取角速度值
  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;
  Eigen::Vector3d angular_velocity(rx, ry, rz);

  Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

  Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
  tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

  Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

  tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
  tmp_V = tmp_V + dt * un_acc;

  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;

  latest_time = t;    // 当前帧变上一帧
}

void update() {
  TicToc t_predict;
  latest_time = current_time;
  tmp_P = estimator.Ps[WINDOW_SIZE];
  tmp_Q = estimator.Rs[WINDOW_SIZE];
  tmp_V = estimator.Vs[WINDOW_SIZE];
  tmp_Ba = estimator.Bas[WINDOW_SIZE];
  tmp_Bg = estimator.Bgs[WINDOW_SIZE];
  acc_0 = estimator.acc_0;
  gyr_0 = estimator.gyr_0;

  queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
  for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
    predict(tmp_imu_buf.front());
}

/**
 * @brief 将图像特征和IMU数据进行对齐
 * 
 * @return std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> IMU数据 和 图像配对 
 */
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements() {
  std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

  while (true) {
    if (imu_buf.empty() || feature_buf.empty())
      return measurements;

    // 对齐标准1  IMU最后一个数据的时间 需要大于 第一个图像特征数据的时间
    if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td)) {
      // 等待更多的 IMU数据
      sum_of_wait++;
      return measurements;
    }

    // 对齐标准2  IMU第一个数据的时间要 需要大于 第一个图像特征数据的时间
    if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td)) {
      // 丢弃图像数据
      ROS_WARN("throw img, only should happen at the beginning");
      feature_buf.pop();
      continue;
    }

    /************ 满足对齐条件 ***********/
    
    sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();  // 获取图像特征数据
    feature_buf.pop();

    // 获取 两帧图像之间的IMU数据
    std::vector<sensor_msgs::ImuConstPtr> IMUs;
    while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td) {
      IMUs.emplace_back(imu_buf.front());
      imu_buf.pop();
    }

    // 这里把下一个IMU数据也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
    IMUs.emplace_back(imu_buf.front());
    if (IMUs.empty()) {
      ROS_WARN("no imu between two image");
    }
    
    measurements.emplace_back(IMUs, img_msg);
  }
  return measurements;
}


/**
 * @brief IMU数据回调函数
 * 
 * @param[in] imu_msg   IMU 数据
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
  if (imu_msg->header.stamp.toSec() <= last_imu_t) {    // 时间戳有问题
    ROS_WARN("imu message in disorder!"); 
    return;
  }
  last_imu_t = imu_msg->header.stamp.toSec();   // 获得当前IMU数据的时间戳

  m_buf.lock();
  imu_buf.push(imu_msg);    // 
  m_buf.unlock();
  con.notify_one();

  last_imu_t = imu_msg->header.stamp.toSec();

  {
    std::lock_guard<std::mutex> lg(m_state);
    predict(imu_msg);
    std_msgs::Header header = imu_msg->header;
    header.frame_id = "world";
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
      pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
  }
}

/**
 * @brief 特征回调函数，用于获取视觉前端提取的特征
 * 
 * @param[in] feature_msg   特征数据
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
  if (!init_feature) {
    // skip the first detected feature, which doesn't contain optical flow speed
    init_feature = 1;
    return;
  }
  m_buf.lock();
  feature_buf.push(feature_msg);
  m_buf.unlock();
  con.notify_one();
}



void restart_callback(const std_msgs::BoolConstPtr &restart_msg) {
  if (restart_msg->data == true) {
    ROS_WARN("restart the estimator!");
    m_buf.lock();
    while (!feature_buf.empty())
      feature_buf.pop();
    while (!imu_buf.empty())
      imu_buf.pop();
    m_buf.unlock();
    m_estimator.lock();
    estimator.clearState();
    estimator.setParameter();
    m_estimator.unlock();
    current_time = -1;
    last_imu_t = 0;
  }
  return;
}

/**
 * @brief 重定位特征回调函数
 *        需要开启在配置文件中开启 fast_relocalization = 1
 * 
 * @param[in] points_msg  重定位观测到的特征
 */
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg) {
  m_buf.lock();
  relo_buf.push(points_msg);
  m_buf.unlock();
}

/**
 * @brief VIO 主线程：
 * Step 1: getMeasurements()        等待并获取同步的IMU数据、特征点数据
 * Step 2: estimator.processIMU()   进行IMU预积分     
 * Step 3: estimator.setReloFrame() 设置重定位帧
 * Step 4: estimator.processImage() 处理图像帧：VIO初始化、后端非线性优化  
 * Step 5: 发布各种数据
 */
void process() {
  while (true) {
    // 数据配对：图像特征 和 两帧之间的 IMU数据配对
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    std::unique_lock<std::mutex> lk(m_buf);
    con.wait(lk, [&] {
      return (measurements = getMeasurements()).size() != 0;    // 获得 配对的特征数据和IMU数据  
    });
    lk.unlock();

    m_estimator.lock();
    // 遍历数据
    for (auto &measurement : measurements) {
      auto img_msg = measurement.second;  // 获得 特征数据
      double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

      /**************************** 处理IMU数据 **********************************/
      // 遍历 IMU数据
      for (auto &imu_msg : measurement.first) {   
        double t = imu_msg->header.stamp.toSec(); // 获得 当前帧IMU时间戳
        double img_t = img_msg->header.stamp.toSec() + estimator.td;  // 获得 图像对应的时间戳
        if (t <= img_t) {
          if (current_time < 0)   // 是否是第一帧图像
            current_time = t;
          double dt = t - current_time;
          ROS_ASSERT(dt >= 0);
          // 获取 IMU 数据
          current_time = t;   // 将 IMU时间戳设为当前时间
          dx = imu_msg->linear_acceleration.x;
          dy = imu_msg->linear_acceleration.y;
          dz = imu_msg->linear_acceleration.z;
          rx = imu_msg->angular_velocity.x;
          ry = imu_msg->angular_velocity.y;
          rz = imu_msg->angular_velocity.z;
          estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        } else {
          // td不为0时，IMU积分的最后一帧IMU数据，时间戳示意图：
          // current_time         img_t              t       
          //       |----> dt_1 <----|----> dt_2 <----|
          double dt_1 = img_t - current_time;
          double dt_2 = t - img_t;
          current_time = img_t;     // 将图像帧的时间戳作为当前时间戳
          ROS_ASSERT(dt_1 >= 0);
          ROS_ASSERT(dt_2 >= 0);
          ROS_ASSERT(dt_1 + dt_2 > 0);

          // 计算权重
          double w1 = dt_2 / (dt_1 + dt_2);
          double w2 = dt_1 / (dt_1 + dt_2);
          // 根据权重，估计出 img_t 时刻的 IMU数据
          dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
          dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
          dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
          rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
          ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
          rz = w1 * rz + w2 * imu_msg->angular_velocity.z;

          // IMU 预积分
          estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        }
      } // IMU 数据处理结束


      /***************************** 处理闭环帧 *******************************/
      sensor_msgs::PointCloudConstPtr relo_msg = NULL;
      // 取出最后一个闭环帧
      while (!relo_buf.empty()) {
        relo_msg = relo_buf.front();
        relo_buf.pop();
      }
      
      if (relo_msg != NULL) {
        // 如果存在闭环帧
        vector<Vector3d> match_points;    // 存储重定位帧观测到的3d点
        double frame_stamp = relo_msg->header.stamp.toSec();    // 获得闭环帧对应的关键帧的时间戳
        // 遍历 闭环帧和对应关键帧共视的特征点
        for (unsigned int i = 0; i < relo_msg->points.size(); i++) {
          // 前2维是特征点在闭环帧的归一化坐标，第3维是特征ID
          Vector3d u_v_id;
          u_v_id.x() = relo_msg->points[i].x;
          u_v_id.y() = relo_msg->points[i].y; 
          u_v_id.z() = relo_msg->points[i].z;
          match_points.push_back(u_v_id);
        }

        // 闭环帧的 t_wb (在前端VIO世界坐标系下)
        Vector3d relo_t(relo_msg->channels[0].values[0], 
                        relo_msg->channels[0].values[1], 
                        relo_msg->channels[0].values[2]);   
        // 闭环帧的 R_wb (在前端VIO世界坐标系下)
        Quaterniond relo_q(relo_msg->channels[0].values[3], 
                           relo_msg->channels[0].values[4], 
                           relo_msg->channels[0].values[5], 
                           relo_msg->channels[0].values[6]); 
        Matrix3d relo_r = relo_q.toRotationMatrix();

        int frame_index;
        frame_index = relo_msg->channels[0].values[7];  // 闭环帧的ID
        estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);   // 将闭环帧传入estimator
      } // 闭环帧信息处理结束


      /***************************** 处理前端特征 *******************************/
      ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

      TicToc t_s;
      map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;   // 特征观测图，图结构 <特征ID, <相机ID，特征归一化坐标、像素坐标、像素速度> >
      // 遍历前端特征
      for (unsigned int i = 0; i < img_msg->points.size(); i++) {
        int v = img_msg->channels[0].values[i] + 0.5;     // +0.5 可以使int赋值时四舍五入
        int feature_id = v / NUM_OF_CAM;    // 获得 特征id
        int camera_id = v % NUM_OF_CAM;     // 获得 是左目还是右目
        // 特征在重定位帧归一化平面的坐标
        double x = img_msg->points[i].x;
        double y = img_msg->points[i].y;
        double z = img_msg->points[i].z;
        // 特征在重定位帧的像素坐标
        double p_u = img_msg->channels[1].values[i];
        double p_v = img_msg->channels[2].values[i];
        // 特征在重定位帧的像素运动速度
        double velocity_x = img_msg->channels[3].values[i];
        double velocity_y = img_msg->channels[4].values[i];
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        image[feature_id].emplace_back(camera_id, xyz_uv_velocity);   // 存储特征观测图
      } // 前端特征处理结束

      estimator.processImage(image, img_msg->header);   // estimator 处理图像数据，后端优化
      
      double whole_t = t_s.toc();
      printStatistics(estimator, whole_t);
      std_msgs::Header header = img_msg->header;
      header.frame_id = "world";

      pubOdometry(estimator, header);     // 发布 /vins_estimator/path 相机轨迹，并记录到输出文件里
      pubKeyPoses(estimator, header);     
      pubCameraPose(estimator, header); 
      pubPointCloud(estimator, header);
      pubTF(estimator, header); 
      pubKeyframe(estimator);             // 发布 上上帧（最新的关键帧）的位姿和观测到的特征点，用于PoseGraph
      if (relo_msg != NULL)
        pubRelocalization(estimator);     // 发布 在滑窗中优化后的闭环帧信息，用于PoseGraph
      //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
    } // 数据处理结束

    m_estimator.unlock();
    m_buf.lock();
    m_state.lock();
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
      update();
    m_state.unlock();
    m_buf.unlock();
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "vins_estimator");    // 初始化 vins_estimator 节点
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  readParameters(n);         // 读取参数
  estimator.setParameter();  // estimator 配置参数

#ifdef EIGEN_DONT_PARALLELIZE
  ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

  ROS_WARN("waiting for image and imu...");

  registerPub(n);   // 定义并初始化发布的信息

  // 定义 vins_estimator 接收的topic
  ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());  // 接收 IMU数据
  ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);               // 接收 前端特征数据
  ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);             // 接收 restart信号
  ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);  // 接收 posegraph的信息

  // 创建VIO主线程
  std::thread measurement_process{process};

  ros::spin();

  return 0;
}
