#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include <visualization_msgs/Marker.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>
#include "keyframe.h"
#include "parameters.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::ImageConstPtr> image_buf;        // 原始图像数据
queue<sensor_msgs::PointCloudConstPtr> point_buf;   // 关键帧观测到的地图点云信息
queue<nav_msgs::Odometry::ConstPtr> pose_buf;       // 关键帧 pose
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = 0;
bool start_flag = 0;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;
int VISUALIZE_IMU_FORWARD;
int LOOP_CLOSURE;
int FAST_RELOCALIZATION;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
ros::Publisher pub_match_img;
ros::Publisher pub_match_points;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_key_odometrys;
ros::Publisher pub_vio_path;
nav_msgs::Path no_loop_path;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;


/**
 * @brief 创建一个新的轨迹序列
 * 创建的序列号从1开始，最多为5。序列为0代表的是预先加载的地图
 */
void new_sequence() {
  printf("new sequence\n");
  sequence++;
  printf("sequence cnt %d \n", sequence);
  if (sequence > 5) {
    ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
    ROS_BREAK();
  }
  posegraph.posegraph_visualization->reset();
  posegraph.publish();
  m_buf.lock();
  while (!image_buf.empty())
    image_buf.pop();
  while (!point_buf.empty())
    point_buf.pop();
  while (!pose_buf.empty())
    pose_buf.pop();
  while (!odometry_buf.empty())
    odometry_buf.pop();
  m_buf.unlock();
}

/**
 * @brief ROS 回调函数，用于读取图像数据
 * 
 * @param[in] image_msg   图像数据
 */
void image_callback(const sensor_msgs::ImageConstPtr &image_msg) {
  //ROS_INFO("image_callback!");
  if (!LOOP_CLOSURE)
    return;
  m_buf.lock();
  image_buf.push(image_msg);
  m_buf.unlock();
  //printf(" image time %f \n", image_msg->header.stamp.toSec());

  // detect unstable camera stream
  if (last_image_time == -1)
    last_image_time = image_msg->header.stamp.toSec();
  else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time) {
    ROS_WARN("image discontinue! detect a new sequence!");
    new_sequence();
  }
  last_image_time = image_msg->header.stamp.toSec();
}

/**
 * @brief ROS回调函数，接收VIO最新的关键帧观测到的点云数据
 * 每个点云对应一个 channel，包含信息如下:
 * [0-1] 特征点在上上帧相机的归一化坐标
 * [2-3] 特征点在上上帧相机的像素坐标
 * [4]   特征 ID
 * 
 * @param[in] point_msg   点云数据
 */
void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg) {
  //ROS_INFO("point_callback!");
  if (!LOOP_CLOSURE)
    return;
  m_buf.lock();
  point_buf.push(point_msg);
  m_buf.unlock();
}

/**
 * @brief ROS回调函数，接收VIO最新的关键帧位姿 T_wb
 * 
 * @param[in] pose_msg  VIO最新的关键帧位姿 T_wb
 */
void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  //ROS_INFO("pose_callback!");
  if (!LOOP_CLOSURE)
    return;
  m_buf.lock();
  pose_buf.push(pose_msg);
  m_buf.unlock();
}


void imu_forward_callback(const nav_msgs::Odometry::ConstPtr &forward_msg) {
  if (VISUALIZE_IMU_FORWARD) {
    Vector3d vio_t(forward_msg->pose.pose.position.x, forward_msg->pose.pose.position.y, forward_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = forward_msg->pose.pose.orientation.w;
    vio_q.x() = forward_msg->pose.pose.orientation.x;
    vio_q.y() = forward_msg->pose.pose.orientation.y;
    vio_q.z() = forward_msg->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio * vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, forward_msg->header);
  }
}

/**
 * @brief ROS回调函数，接收在滑窗内优化过的 闭环帧和对应关键帧的相对位姿 T_loop_cur
 * 
 * @param[in] pose_msg T_loop_cur
 */
void relo_relative_pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  Vector3d relative_t = Vector3d(pose_msg->pose.pose.position.x,
                                 pose_msg->pose.pose.position.y,
                                 pose_msg->pose.pose.position.z);
  Quaterniond relative_q;
  relative_q.w() = pose_msg->pose.pose.orientation.w;
  relative_q.x() = pose_msg->pose.pose.orientation.x;
  relative_q.y() = pose_msg->pose.pose.orientation.y;
  relative_q.z() = pose_msg->pose.pose.orientation.z;
  double relative_yaw = pose_msg->twist.twist.linear.x;
  int index = pose_msg->twist.twist.linear.y;
  //printf("receive index %d \n", index );
  Eigen::Matrix<double, 8, 1> loop_info;
  loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
      relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
      relative_yaw;
  posegraph.updateKeyFrameLoop(index, loop_info);   // 更新 T_loop_cur
}


void vio_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  //ROS_INFO("vio_callback!");
  Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
  Quaterniond vio_q;
  vio_q.w() = pose_msg->pose.pose.orientation.w;
  vio_q.x() = pose_msg->pose.pose.orientation.x;
  vio_q.y() = pose_msg->pose.pose.orientation.y;
  vio_q.z() = pose_msg->pose.pose.orientation.z;

  vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
  vio_q = posegraph.w_r_vio * vio_q;

  vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
  vio_q = posegraph.r_drift * vio_q;

  Vector3d vio_t_cam;
  Quaterniond vio_q_cam;
  vio_t_cam = vio_t + vio_q * tic;
  vio_q_cam = vio_q * qic;

  if (!VISUALIZE_IMU_FORWARD) {
    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
  }

  odometry_buf.push(vio_t_cam);
  if (odometry_buf.size() > 10) {
    odometry_buf.pop();
  }

  visualization_msgs::Marker key_odometrys;
  key_odometrys.header = pose_msg->header;
  key_odometrys.header.frame_id = "world";
  key_odometrys.ns = "key_odometrys";
  key_odometrys.type = visualization_msgs::Marker::SPHERE_LIST;
  key_odometrys.action = visualization_msgs::Marker::ADD;
  key_odometrys.pose.orientation.w = 1.0;
  key_odometrys.lifetime = ros::Duration();

  //static int key_odometrys_id = 0;
  key_odometrys.id = 0;  //key_odometrys_id++;
  key_odometrys.scale.x = 0.1;
  key_odometrys.scale.y = 0.1;
  key_odometrys.scale.z = 0.1;
  key_odometrys.color.r = 1.0;
  key_odometrys.color.a = 1.0;

  for (unsigned int i = 0; i < odometry_buf.size(); i++) {
    geometry_msgs::Point pose_marker;
    Vector3d vio_t;
    vio_t = odometry_buf.front();
    odometry_buf.pop();
    pose_marker.x = vio_t.x();
    pose_marker.y = vio_t.y();
    pose_marker.z = vio_t.z();
    key_odometrys.points.push_back(pose_marker);
    odometry_buf.push(vio_t);
  }
  pub_key_odometrys.publish(key_odometrys);

  if (!LOOP_CLOSURE) {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = pose_msg->header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = vio_t.x();
    pose_stamped.pose.position.y = vio_t.y();
    pose_stamped.pose.position.z = vio_t.z();
    no_loop_path.header = pose_msg->header;
    no_loop_path.header.frame_id = "world";
    no_loop_path.poses.push_back(pose_stamped);
    pub_vio_path.publish(no_loop_path);
  }
}

/**
 * @brief ROS 回调函数，接收IMU和相机之间的外参 T_bc
 * 
 * @param[in] pose_msg  IMU和相机之间的外参 T_bc
 */
void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg) {
  m_process.lock();
  tic = Vector3d(pose_msg->pose.pose.position.x,
                 pose_msg->pose.pose.position.y,
                 pose_msg->pose.pose.position.z);
  qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                    pose_msg->pose.pose.orientation.x,
                    pose_msg->pose.pose.orientation.y,
                    pose_msg->pose.pose.orientation.z)
            .toRotationMatrix();
  m_process.unlock();
}


/**
 * @brief PoseGraph 主线程
 */
void process() {
  if (!LOOP_CLOSURE)
    return;
  while (true) {
    sensor_msgs::ImageConstPtr image_msg = NULL;
    sensor_msgs::PointCloudConstPtr point_msg = NULL;
    nav_msgs::Odometry::ConstPtr pose_msg = NULL;

    // 得到具有相同时间戳的 pose_msg、image_msg、point_msg
    m_buf.lock();
    if (!image_buf.empty() && !point_buf.empty() && !pose_buf.empty()) {
      if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec()) { // 丢弃早于图像是位姿信息
        pose_buf.pop(); 
        printf("throw pose at beginning\n");
      } else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec()) { // 丢弃早于图像的特征信息
        point_buf.pop();
        printf("throw point at beginning\n");
      } else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec()) {
        pose_msg = pose_buf.front();    // 获得最新的关键帧位姿
        pose_buf.pop();
        while (!pose_buf.empty())
          pose_buf.pop();
        while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())  // 丢弃早于关键帧位姿的图像信息
          image_buf.pop();
        image_msg = image_buf.front();  // 获得图像信息 
        image_buf.pop();

        while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())  // 丢弃遭遇关键帧位姿的特征信息
          point_buf.pop();
        point_msg = point_buf.front();  // 获得关键帧观测到的特征信息
        point_buf.pop();
      }
    }
    m_buf.unlock();

    if (pose_msg != NULL) {
      // 不考虑最开始的几帧
      if (skip_first_cnt < SKIP_FIRST_CNT) {
        skip_first_cnt++;
        continue;
      }

      // 每隔SKIP_CNT帧进行一次回环检测，默认 SKIP_CNT=0
      if (skip_cnt < SKIP_CNT) {
        skip_cnt++;
        continue;
      } else {
        skip_cnt = 0;
      }

      // 将 ROS 的图像数据转换为 opencv 的数据
      cv_bridge::CvImageConstPtr ptr;
      if (image_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = image_msg->header;
        img.height = image_msg->height;
        img.width = image_msg->width;
        img.is_bigendian = image_msg->is_bigendian;
        img.step = image_msg->step;
        img.data = image_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      } else
        ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

      cv::Mat image = ptr->image;
      // 转换 当前关键帧的位姿 T_wb
      // t_wb
      Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                            pose_msg->pose.pose.position.y,
                            pose_msg->pose.pose.position.z);        
      
      // R_wb
      Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                               pose_msg->pose.pose.orientation.x,
                               pose_msg->pose.pose.orientation.y,
                               pose_msg->pose.pose.orientation.z) 
                       .toRotationMatrix();     

      // 运动距离大于SKIP_DIS才进行回环检测，SKIP_DIS默认为0           
      if ((T - last_t).norm() > SKIP_DIS) {
        vector<cv::Point3f> point_3d;
        vector<cv::Point2f> point_2d_uv;
        vector<cv::Point2f> point_2d_normal;
        vector<double> point_id;

         // 遍历当前关键帧观测到的所有特征
        for (unsigned int i = 0; i < point_msg->points.size(); i++) {
          cv::Point3f p_3d;     
          // 特征点的世界坐标
          p_3d.x = point_msg->points[i].x;    
          p_3d.y = point_msg->points[i].y;    
          p_3d.z = point_msg->points[i].z;
          point_3d.push_back(p_3d);

          cv::Point2f p_2d_uv, p_2d_normal;
          double p_id;

          // 特征点在相机归一化平面的坐标
          p_2d_normal.x = point_msg->channels[i].values[0];
          p_2d_normal.y = point_msg->channels[i].values[1];
          point_2d_normal.push_back(p_2d_normal);

          // 特征点的像素坐标
          p_2d_uv.x = point_msg->channels[i].values[2];
          p_2d_uv.y = point_msg->channels[i].values[3];
          point_2d_uv.push_back(p_2d_uv);
          
          // 特征点id
          p_id = point_msg->channels[i].values[4];
          point_id.push_back(p_id);
        }

        // 创建关键帧
        KeyFrame *keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), frame_index, T, R, image,
                                          point_3d, point_2d_uv, point_2d_normal, point_id, sequence);
        m_process.lock();
        start_flag = 1;
        posegraph.addKeyFrame(keyframe, 1);     // 向 PoseGraph 添加关键帧，开始闭环检测
        m_process.unlock();
        frame_index++;
        last_t = T;
      }
    }

    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

/**
 * @brief 键盘控制线程，用于保存posegraph 或者 创建新的序列
 */
void command() {
  if (!LOOP_CLOSURE)
    return;
  while (1) {
    char c = getchar();
    if (c == 's') {
      m_process.lock();
      posegraph.savePoseGraph();  // 保存 posegraph
      m_process.unlock();
      printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
      // printf("program shutting down...\n");
      // ros::shutdown();
    }
    if (c == 'n')
      new_sequence();   // 创建一个新的轨迹序列

    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char **argv) {
  // ROS初始化
  ros::init(argc, argv, "pose_graph");
  ros::NodeHandle n("~");
  posegraph.registerPub(n);

  // 读取参数
  n.getParam("visualization_shift_x", VISUALIZATION_SHIFT_X);
  n.getParam("visualization_shift_y", VISUALIZATION_SHIFT_Y);
  n.getParam("skip_cnt", SKIP_CNT);   
  n.getParam("skip_dis", SKIP_DIS); 
  std::string config_file;
  n.getParam("config_file", config_file);
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  double camera_visual_size = fsSettings["visualize_camera_size"];
  cameraposevisual.setScale(camera_visual_size);
  cameraposevisual.setLineWidth(camera_visual_size / 10.0);

  LOOP_CLOSURE = fsSettings["loop_closure"];
  std::string IMAGE_TOPIC;
  int LOAD_PREVIOUS_POSE_GRAPH;

  // 如果需要进行回环
  if (LOOP_CLOSURE) {
    ROW = fsSettings["image_height"];   
    COL = fsSettings["image_width"];
    std::string pkg_path = ros::package::getPath("pose_graph");
    string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin";
    cout << "vocabulary_file" << vocabulary_file << endl;
    posegraph.loadVocabulary(vocabulary_file);    // 加载词典文件

    // BRIEF 描述子 pattern 样式文件路径
    BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml";
    cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config_file.c_str());

    fsSettings["image_topic"] >> IMAGE_TOPIC;                     // 图像Topic名字
    fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;   // posegraph路径
    fsSettings["output_path"] >> VINS_RESULT_PATH;                // 位姿结果输出路径 
    fsSettings["save_image"] >> DEBUG_IMAGE;

    // create folder if not exists
    FileSystemHelper::createDirectoryIfNotExists(POSE_GRAPH_SAVE_PATH.c_str());
    FileSystemHelper::createDirectoryIfNotExists(VINS_RESULT_PATH.c_str());

    VISUALIZE_IMU_FORWARD = fsSettings["visualize_imu_forward"];
    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"];
    FAST_RELOCALIZATION = fsSettings["fast_relocalization"];
    VINS_RESULT_PATH = VINS_RESULT_PATH + "/vins_result_loop.txt";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();
    fsSettings.release();

    // 是否加载已经存在的地图
    if (LOAD_PREVIOUS_POSE_GRAPH) {
      printf("load pose graph\n");
      m_process.lock();
      posegraph.loadPoseGraph();      // 加载已有的地图
      m_process.unlock();
      printf("load pose graph finish\n");
      load_flag = 1;
    } else {
      printf("no previous pose graph\n");
      load_flag = 1;
    }
  }

  fsSettings.release();

  // ros接收的数据
  ros::Subscriber sub_imu_forward = n.subscribe("/vins_estimator/imu_propagate", 2000, imu_forward_callback);
  ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 2000, vio_callback);                       
  ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2000, image_callback);                           // 接收 图像数据
  ros::Subscriber sub_pose = n.subscribe("/vins_estimator/keyframe_pose", 2000, pose_callback);         // 接收 VIO 上上帧（最新的关键帧）的位姿
  ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 2000, extrinsic_callback);   // 接收 相机和IMU之间的外参
  ros::Subscriber sub_point = n.subscribe("/vins_estimator/keyframe_point", 2000, point_callback);      // 接收 VIO 上上帧（最新的关键帧）观测到的所有特征
  ros::Subscriber sub_relo_relative_pose = n.subscribe("/vins_estimator/relo_relative_pose", 2000, relo_relative_pose_callback);    // 接收 在滑窗中优化后的闭环帧位姿

  // ros发布的数据
  pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
  pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
  pub_key_odometrys = n.advertise<visualization_msgs::Marker>("key_odometrys", 1000);
  pub_vio_path = n.advertise<nav_msgs::Path>("no_loop_path", 1000);
  pub_match_points = n.advertise<sensor_msgs::PointCloud>("match_points", 100);

  std::thread measurement_process;
  std::thread keyboard_command_process;

  // pose graph主线程
  measurement_process = std::thread(process);
  
  // 键盘控制线程
  keyboard_command_process = std::thread(command);

  ros::spin();

  return 0;
}
