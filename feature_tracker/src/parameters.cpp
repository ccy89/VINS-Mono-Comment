#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name) {
  T ans;
  if (n.getParam(name, ans)) {
    ROS_INFO_STREAM("Loaded " << name << ": " << ans);
  } else {
    ROS_ERROR_STREAM("Failed to load " << name);
    n.shutdown();
  }
  return ans;
}

void readParameters(ros::NodeHandle &n) {
  std::string config_file;
  config_file = readParam<std::string>(n, "config_file");          // 获取配置文件路径
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);  // 读取yaml参数文件
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }
  std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

  fsSettings["image_topic"] >> IMAGE_TOPIC; // 单目图像 topic name
  fsSettings["imu_topic"] >> IMU_TOPIC;     // IMU数据 topic name
  MAX_CNT = fsSettings["max_cnt"];          // 单张图像最多提取的特征点数目
  MIN_DIST = fsSettings["min_dist"];        // 特征点之间的最小距离
  ROW = fsSettings["image_height"];         // 图像高度
  COL = fsSettings["image_width"];          // 图像宽度
  FREQ = fsSettings["freq"];                // 轨迹发布频率
  F_THRESHOLD = fsSettings["F_threshold"];  // 基础矩阵求解时的阈值
  SHOW_TRACK = fsSettings["show_track"];    // 是否显示光流跟踪结果
  EQUALIZE = fsSettings["equalize"];        // 是否使用图像增强，如果图片太暗
  FISHEYE = fsSettings["fisheye"];          // 相机是否是鱼眼相机
  if (FISHEYE == 1)
    FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";    // 读取 鱼眼相机的 mask
  CAM_NAMES.push_back(config_file);

  WINDOW_SIZE = 20;         
  STEREO_TRACK = false;     
  FOCAL_LENGTH = 460;       
  PUB_THIS_FRAME = false;   

  if (FREQ == 0)
    FREQ = 100;

  fsSettings.release();
}
