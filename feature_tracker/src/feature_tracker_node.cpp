#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img, pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];   // 特征跟踪器
double first_image_time;                  // 
int pub_count = 1;                        
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;


/**
 * @brief ROS 接收图像数据的回调函数，对接收到的图像数据进行光流跟踪，并发布跟踪到特征
 *   
 * @param[in] img_msg   
 */
void img_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  if (first_image_flag) {   
    // 记录第一张图像
    first_image_flag = false;
    first_image_time = img_msg->header.stamp.toSec();
    last_image_time = img_msg->header.stamp.toSec();
    return;
  }

  // 通过时间间隔判断相机数据流是否稳定，有问题则restart
  if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time) {
    ROS_WARN("image discontinue! reset the feature tracker!");
    // 重置跟踪状态
    first_image_flag = true;
    last_image_time = 0;
    pub_count = 1;
    std_msgs::Bool restart_flag;
    restart_flag.data = true;
    pub_restart.publish(restart_flag);    // 发出 restart信号
    return;
  }
  last_image_time = img_msg->header.stamp.toSec();

  // 控制发布频率
  if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ) {    // 发布频率 <= 设定值
    PUB_THIS_FRAME = true;
    // 时间间隔内的发布频率十分接近设定频率时，更新时间间隔起始时刻，并将数据发布次数置0
    if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ) {
      first_image_time = img_msg->header.stamp.toSec();
      pub_count = 0;
    }
  } else {    // 发布频率 > 设定值
    PUB_THIS_FRAME = false;   // 不发布当前帧
  }
    

  cv_bridge::CvImageConstPtr ptr;
  // 将图像编码 8UC1 转换为 mono8
  if (img_msg->encoding == "8UC1") {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  } else
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  cv::Mat show_img = ptr->image;
  TicToc t_r;
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ROS_DEBUG("processing camera %d", i);
    if (i != 1 || !STEREO_TRACK) {  // 单目
      trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());   // 读取图像并跟踪特征
    } else {  // 双目
      if (EQUALIZE) {
        // 自适应直方图均衡化处理
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();   
        clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
      } else
        trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
    }

#if SHOW_UNDISTORTION
    trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
  }

  // 更新全局ID
  for (unsigned int i = 0;; i++) {
    bool completed = false;
    for (int j = 0; j < NUM_OF_CAM; j++)
      if (j != 1 || !STEREO_TRACK)
        completed |= trackerData[j].updateID(i);
    if (!completed)
      break;
  }

  // 发布向前帧图像跟踪到的特征点
  // 将特征点的id，矫正后归一化平面坐标(x,y,z=1)，像素坐标(u,v)，像素速度(vx,vy)， 封装成sensor_msgs::PointCloudPtr 类型的 feature_points 实例中
  if (PUB_THIS_FRAME) {
    pub_count++;
    sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);   // ROS PointCloud 数据类型
    sensor_msgs::ChannelFloat32 id_of_point;    // 特征id
    sensor_msgs::ChannelFloat32 u_of_point;     // 像素坐标u
    sensor_msgs::ChannelFloat32 v_of_point;     // 像素坐标v
    sensor_msgs::ChannelFloat32 velocity_x_of_point;  // 像素速度u
    sensor_msgs::ChannelFloat32 velocity_y_of_point;  // 像素速度v

    feature_points->header = img_msg->header;
    feature_points->header.frame_id = "world";

    vector<set<int>> hash_ids(NUM_OF_CAM);
    for (int i = 0; i < NUM_OF_CAM; i++) {
      auto &un_pts = trackerData[i].cur_un_pts;
      auto &cur_pts = trackerData[i].cur_pts;
      auto &ids = trackerData[i].ids;
      auto &pts_velocity = trackerData[i].pts_velocity;

      // 遍历 所有跟踪到的特征
      for (unsigned int j = 0; j < ids.size(); j++) {
        if (trackerData[i].track_cnt[j] > 1) {
          int p_id = ids[j];          // 获得 特征id
          hash_ids[i].insert(p_id);
          geometry_msgs::Point32 p;   // 归一化平面坐标
          p.x = un_pts[j].x;
          p.y = un_pts[j].y;          
          p.z = 1;  

          feature_points->points.push_back(p);    // 点云3D坐标就是特征在归一化平面的坐标
          id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
          u_of_point.values.push_back(cur_pts[j].x);
          v_of_point.values.push_back(cur_pts[j].y);
          velocity_x_of_point.values.push_back(pts_velocity[j].x);
          velocity_y_of_point.values.push_back(pts_velocity[j].y);
        }
      }
    }
    // 生成 点云数据
    feature_points->channels.push_back(id_of_point);
    feature_points->channels.push_back(u_of_point);
    feature_points->channels.push_back(v_of_point);
    feature_points->channels.push_back(velocity_x_of_point);
    feature_points->channels.push_back(velocity_y_of_point);
    ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
    // skip the first image; since no optical speed on frist image
    if (!init_pub) {  // 第1帧不发布
      init_pub = 1;
    } else
      pub_img.publish(feature_points);  // 发布特征

    if (SHOW_TRACK) {
      ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
      cv::Mat stereo_img = ptr->image;

      for (int i = 0; i < NUM_OF_CAM; i++) {
        cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
        // 显示所有特征的跟踪状态，越红越好，越蓝越不行
        for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++) {
          double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
          cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
      }
      pub_match.publish(ptr->toImageMsg());
    }
  }
  ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "feature_tracker");     // 初始化 feature_tracker 节点
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  readParameters(n);

  // 读取 每个相机对应的相机内参
  for (int i = 0; i < NUM_OF_CAM; i++)
    trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

  // 对于鱼眼相机，需要用 mask 来去除边界
  if (FISHEYE) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
      trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
      if (!trackerData[i].fisheye_mask.data) {
        ROS_INFO("load mask fail");
        ROS_BREAK();
      } else
        ROS_INFO("load mask success");
    }
  }

  ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);    // 接收 图像数据

  pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);    // 发布 当前图像跟踪到的特征点
  pub_match = n.advertise<sensor_msgs::Image>("feature_img", 1000);   // 发布 带有特征图，用于 Rviz显示
  pub_restart = n.advertise<std_msgs::Bool>("restart", 1000);         // 发布 restart信号

  ros::spin();
  return 0;
}

// new points velocity is 0, pub or not?
// track cnt > 1 pub?