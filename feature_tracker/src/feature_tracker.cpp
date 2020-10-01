#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

/**
 * @brief 判断像素点是否落在图像边缘
 * 
 * @param[in] pt  像素坐标
 * @return true   像素点不在图像边界
 * @return false  像素点在图像边界
 */
bool inBorder(const cv::Point2f &pt) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 根据 status  剔除向量中的变量
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

// 根据 status  剔除向量中的变量
void reduceVector(vector<int> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}


FeatureTracker::FeatureTracker() {
}

/**
 * @brief 剔除密集的特征点，保证特征点之间距离至少为30
 */
void FeatureTracker::setMask() {
  if (FISHEYE)
    mask = fisheye_mask.clone();
  else
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

  // 构造 特征跟踪序列，序列结构 <特征点跟踪成功次数，<特征点像素坐标，特征点id>>
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
  for (unsigned int i = 0; i < forw_pts.size(); i++)
    cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

  // 根据特征跟踪的次数进行排序
  // 优先保留 跟踪次数多的点
  sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
    return a.first > b.first;
  });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  // 遍历特征，挑选特征点
  for (auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {   // 判断 mask
      forw_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      // 将 距离当前特征点 30个像素的区域标记为-1，落在该区域内的特征点将被剔除
      cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
    }
  }
}

// 添将新检测到的特征点n_pts
void FeatureTracker::addPoints() {
  for (auto &p : n_pts) {
    forw_pts.push_back(p);
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

/**
 * @brief 读取图像并跟踪特征
 * 
 * @param[in] _img        图像          
 * @param[in] _cur_time   图像时间戳
 */
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time) {
  cv::Mat img;
  TicToc t_r;
  cur_time = _cur_time;

  // 如果EQUALIZE=1，表示太亮或太暗，进行直方图均衡化处理
  if (EQUALIZE) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    TicToc t_c;
    clahe->apply(_img, img);
    ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
  } else
    img = _img;

  if (forw_img.empty()) {
    // 如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据
    // 将读入的图像赋给当前帧forw_img，同时还赋给prev_img、cur_img
    prev_img = cur_img = forw_img = img;
  } else {
    // 否则，说明之前就已经有图像读入，只需要更新当前帧forw_img的数据
    forw_img = img;
  }

  forw_pts.clear();

  if (cur_pts.size() > 0) {
    TicToc t_o;
    vector<uchar> status;
    vector<float> err;

    // 光流跟踪，对前一帧的特征点cur_pts进行LK金字塔光流跟踪，得到forw_pts
    // status标记了从前一帧cur_img到forw_img特征点的跟踪状态，无法被追踪到的点标记为0
    // 每个金字塔层的搜索窗口大小为 (21,21)
    // 图像金字塔为4层
    // 默认跟踪收敛条件为 迭代次数+误差大小，最大迭代次数为30，误差阈值为0.01
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

    // 剔除位于图像边界的点
    for (int i = 0; i < int(forw_pts.size()); i++)
      if (status[i] && !inBorder(forw_pts[i]))
        status[i] = 0;
    
    // 根据status，从 prev_pts、cur_pts 和 forw_pts 剔除不良特征
    // prev_pts 和 cur_pts 中的特征点是一一对应的
    // 记录特征点id的 ids，和记录特征点被跟踪次数的 track_cnt 也要剔除
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
  }

  // 跟踪成功的特征，跟踪次数+1
  for (auto &n : track_cnt)
    n++;

  if (PUB_THIS_FRAME) { // 如果当前帧需要发布  
    // 通过基本矩阵剔除 outliers
    rejectWithF();      
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    setMask();  // 保证相邻的特征点之间要相隔30个像素，设置mask
    ROS_DEBUG("set mask costs %fms", t_m.toc());

    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());  // 计算是否需要提取新的特征
    if (n_max_cnt > 0) {
      if (mask.empty())
        cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1)
        cout << "mask type wrong " << endl;
      if (mask.size() != forw_img.size())
        cout << "wrong size " << endl;

      /** 
        *void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
        *   InputArray  image,              输入图像
        *   OutputArray   corners,          存放检测到的角点的vector
        *   int     maxCorners,             返回的角点的数量的最大值
        *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
        *   double  minDistance,            返回角点之间欧式距离的最小值
        *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
        *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
        *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
        *   double  k = 0.04                Harris角点检测需要的k值
        *)   
        */
      cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
    } else
      n_pts.clear();
    ROS_DEBUG("detect feature costs: %fms", t_t.toc());

    ROS_DEBUG("add feature begins");
    TicToc t_a;
    // 将新检测到的特征点 n_pts 添加到 forw_pts中，id 初始化-1，track_cnt初始化为1.
    addPoints();
    ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
  }
  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  cur_img = forw_img;
  cur_pts = forw_pts;
  undistortedPoints();
  prev_time = cur_time;
}


/**
 * @brief 利用基础矩阵剔除外点
 */
void FeatureTracker::rejectWithF() {
  if (forw_pts.size() >= 8) {
    ROS_DEBUG("FM ransac begins");
    TicToc t_f;
    vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
    
    // 遍历所有匹配的特征对 
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      Eigen::Vector3d tmp_p;
      // 根据相机模型 上一帧特征像素的坐标转换到相机坐标系下（包含畸变矫正），并转换为归一化像素坐标
      m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      // 根据相机模型将 当前帧特征的像素坐标转换到相机坐标系下（包含畸变矫正），并转换为归一化像素坐标
      m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);  // 计算 F矩阵
    int size_a = cur_pts.size();
    // 剔除外点
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
    ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

bool FeatureTracker::updateID(unsigned int i) {
  if (i < ids.size()) {
    if (ids[i] == -1)
      ids[i] = n_id++;
    return true;
  } else
    return false;
}


void FeatureTracker::readIntrinsicParameter(const string &calib_file) {
  ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name) {
  cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < COL; i++)
    for (int j = 0; j < ROW; j++) {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
      //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }
  for (int i = 0; i < int(undistortedp.size()); i++) {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
    pp.at<float>(2, 0) = 1.0;

    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600) {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    } else {
      //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  cv::imshow(name, undistortedImg);
  cv::waitKey(0);
}

void FeatureTracker::undistortedPoints() {
  cur_un_pts.clear();
  cur_un_pts_map.clear();
  //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
  for (unsigned int i = 0; i < cur_pts.size(); i++) {
    Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    m_camera->liftProjective(a, b);
    cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
    //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
  }
  // caculate points velocity
  if (!prev_un_pts_map.empty()) {
    double dt = cur_time - prev_time;
    pts_velocity.clear();
    for (unsigned int i = 0; i < cur_un_pts.size(); i++) {
      if (ids[i] != -1) {
        std::map<int, cv::Point2f>::iterator it;
        it = prev_un_pts_map.find(ids[i]);
        if (it != prev_un_pts_map.end()) {
          double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.push_back(cv::Point2f(v_x, v_y));
        } else
          pts_velocity.push_back(cv::Point2f(0, 0));
      } else {
        pts_velocity.push_back(cv::Point2f(0, 0));
      }
    }
  } else {
    for (unsigned int i = 0; i < cur_pts.size(); i++) {
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}
