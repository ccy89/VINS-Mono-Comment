#include "feature_manager.h"

// 获得 当前特征点在滑窗内的最后一帧观测的索引
int FeaturePerId::endFrame() {
  return start_frame + feature_per_frame.size() - 1;
}


FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs) {
  for (int i = 0; i < NUM_OF_CAM; i++)
    ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[]) {
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ric[i] = _ric[i];
  }
}

void FeatureManager::clearState() {
  feature.clear();
}

int FeatureManager::getFeatureCount() {
  int cnt = 0;
  for (auto &it : feature) {
    it.used_num = it.feature_per_frame.size();

    if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2) {
      cnt++;
    }
  }
  return cnt;
}


/**
 * @brief 计算每一个特征点的跟踪次数 和 它在上一帧和上上帧间的视差，判断上一帧是否是关键帧
 * 
 * @param[in] frame_count   当前图像帧在滑动窗口内的索引
 * @param[in] image         图像观测到的所有特征点，数据结构为 <特征点id，<相机id，特征点归一化坐标、特征点像素坐标、特征点像素速度 > >
 * @param[in] td            IMU 和 相机 同步时间差
 * 
 * @return true     上一帧是关键帧; 
 * @return false    上一帧不是关键帧
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td) {
  ROS_DEBUG("input feature: %d", (int)image.size());
  ROS_DEBUG("num of feature: %d", getFeatureCount());
  double parallax_sum = 0;  // 两帧图像的总视察
  int parallax_num = 0;     //
  last_track_num = 0;       // 当前帧图像跟踪到的特征点数目

  // 遍历当前帧观测到的所有特征，
  // 统计跟踪成功的特征数目，并将新提取到的特征添加进 feature list 
  for (auto &id_pts : image) {
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td);  // 构造 当前特征点的数据结构

    int feature_id = id_pts.first;    // 获得特征ID
    // 寻找 feature list 中是否有当前特征点， it 为当前特征点的 feature_id 在 feature list 中的迭代器 
    auto it = find_if(feature.begin(), feature.end(),
                      [feature_id](const FeaturePerId &it) {
                        return it.feature_id == feature_id;
                      }
    );

    if (it == feature.end()) {
      // 在 feature list 中没有找到当前特征点，将当前特征添加进 feature list 
      feature.push_back(FeaturePerId(feature_id, frame_count));   
      feature.back().feature_per_frame.push_back(f_per_fra);
    } else if (it->feature_id == feature_id) {
      // 在feature list 中找到了当前特征点
      it->feature_per_frame.push_back(f_per_fra);   // 存储观测信息
      last_track_num++;
    }
  } // 遍历特征结束

  // 如果当前图像为 滑动窗口起始帧2帧 | 当前图像跟踪到的特征点数目 < 20
  // 认为上一帧图像为关键帧
  if (frame_count < 2 || last_track_num < 20)
    return true;

  // 遍历 feature list
  for (auto &it_per_id : feature) {
    // 判断上一帧和上上帧图像 是否都观测到 当前帧图像的特征点
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) {
      parallax_sum += compensatedParallax2(it_per_id, frame_count);   // 累加特征点的视差
      parallax_num++;
    }
  }

  if (parallax_num == 0) {  
    // 如果当前帧图像特征都是新观测到的，则认为上一帧是关键帧
    return true;
  } else {
    ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
    ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
    return parallax_sum / parallax_num >= MIN_PARALLAX;
  }
}

void FeatureManager::debugShow() {
  ROS_DEBUG("debug show");
  for (auto &it : feature) {
    ROS_ASSERT(it.feature_per_frame.size() != 0);
    ROS_ASSERT(it.start_frame >= 0);
    ROS_ASSERT(it.used_num >= 0);

    ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
    int sum = 0;
    for (auto &j : it.feature_per_frame) {
      ROS_DEBUG("%d,", int(j.is_used));
      sum += j.is_used;
      printf("(%lf,%lf) ", j.point(0), j.point(1));
    }
    ROS_ASSERT(it.used_num == sum);
  }
}

/**
 * @brief 找到两个图像帧共视的特征点
 * 
 * @param[in] frame_count_l   图像帧l的索引
 * @param[in] frame_count_r   图像帧r的索引    r >= l
 * 
 * @return vector<pair<Vector3d, Vector3d>>   // 共视关键帧点对，坐标是归一化平面的坐标
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) {
  vector<pair<Vector3d, Vector3d>> corres;    
  // 遍历 当前滑窗内所有的特征
  for (auto &it : feature) {
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {  // 满足共视
      Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();  
      int idx_l = frame_count_l - it.start_frame;   // 
      int idx_r = frame_count_r - it.start_frame;   // 

      a = it.feature_per_frame[idx_l].point;  
      b = it.feature_per_frame[idx_r].point; 

      corres.push_back(make_pair(a, b));
    }
  }
  return corres;
}

/**
 * @brief 更新后端优化后点的深度
 * 
 * @param[in] x   优化后当前滑窗内所有特征的深度
 */
void FeatureManager::setDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
    //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
    if (it_per_id.estimated_depth < 0) {
      it_per_id.solve_flag = 2;
    } else
      it_per_id.solve_flag = 1;
  }
}


/**
 * @brief 移除外点
 */
void FeatureManager::removeFailures() {
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;
    if (it->solve_flag == 2)
      feature.erase(it);      // 移除外点
  }
}


void FeatureManager::clearDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth = 1.0 / x(++feature_index);
  }
}

/**
 * @brief 
 * 
 * @return VectorXd 
 */
VectorXd FeatureManager::getDepthVector() {
  VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
#if 1
    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
    dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
  }
  return dep_vec;
}

/**
 * @brief 三角化求解特征深度，采用多视图SVD分解的方式
 * 
 * @param[in] Ps    滑窗中所有相机的平移量
 * @param[in] tic   IMU和相机之间的外参 t_bc
 * @param[in] ric   IMU和相机之间的外参 R_bc
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]) {
  // 遍历当前滑窗中的所有特征
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) {  // 如果特征跟踪次数不满2次，无法进行三角化
      continue;
    }

    if (it_per_id.estimated_depth > 0) {  // 已经三角化过的点不再三角化
      continue;
    }
      
    int imu_i = it_per_id.start_frame;    // 滑窗中第一个观测到当前特征的相机
    int imu_j = imu_i - 1;

    ROS_ASSERT(NUM_OF_CAM == 1);
    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;

    // R0 t0为第i帧相机坐标系到世界坐标系的变换矩阵
    Eigen::Matrix<double, 3, 4> P0;
    Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];  // t_wci = R_wbi * t_bc + t_wbi
    Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];              // R_wci = R_wbi * R_bc
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    // 遍历当前特征在滑窗中的观测
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      // R、t为 第j帧相机坐标系 到 第i帧相机坐标系 的变换矩阵，P为i到j的变换矩阵
      Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];  // t_wcj = R_wbj * t_bc + t_wbj
      Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];              // R_wcj = R_wbj * R_bc
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);       // t_cicj = R_wci^T * (t_wcj - t_wci)
      Eigen::Matrix3d R = R0.transpose() * R1;              // R_cicj = R_wci^T * R_wcj
      Eigen::Matrix<double, 3, 4> P;      // 投影矩阵，T_cjci
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized();
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

      if (imu_i == imu_j)
        continue;
    }
    ROS_ASSERT(svd_idx == svd_A.rows());

    // 对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
    Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
    double svd_method = svd_V[2] / svd_V[3];    // SVD 分解计算当前特征点在第i帧相机下的深度

    it_per_id.estimated_depth = svd_method;
    if (it_per_id.estimated_depth < 0.1) {
      it_per_id.estimated_depth = INIT_DEPTH;
    }
  }
}

/**
 * @brief 丢弃特征观测队列中的outlier
 * 这个函数没有用到
 * 
 * @param[in] outlierIndex   所有outlier的ID
 */
void FeatureManager::removeOutlier() {
  ROS_BREAK();
  int i = -1;
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    i += it->used_num != 0;
    if (it->used_num != 0 && it->is_outlier == true) {
      feature.erase(it);
    }
  }
}

/**
 * @brief （系统已经初始化完成）边缘化最老帧时，删除最老帧的观测，传递观测量
 * 
 * @param[in] marg_R  最老帧的 Rwc
 * @param[in] marg_P  最老帧的 twc
 * @param[in] new_R   第2帧的 Rwc
 * @param[in] new_P   第2帧的 twc
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() < 2) {
        feature.erase(it);
        continue;
      } else {
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
        double dep_j = pts_j(2);
        if (dep_j > 0)
          it->estimated_depth = dep_j;
        else
          it->estimated_depth = INIT_DEPTH;
      }
    }
  }
}

/**
 * @brief （系统未完成初始化）边缘化最老帧时，删除最老帧的观测
 */
void FeatureManager::removeBack() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0)
        feature.erase(it);
    }
  }
}

/**
 * @brief 边缘化上一帧时，对特征点在上一帧的信息进行移除处理
 * 
 * @param[in] frame_count 滑窗中图像帧个数
 */
void FeatureManager::removeFront(int frame_count) {
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame == frame_count) {
      it->start_frame--;
    } else {
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endFrame() < frame_count - 1)
        continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0)
        feature.erase(it);
    }
  }
}

/**
 * @brief 
 * 
 * @param[in] it_per_id     某个特征点在滑窗中的所有观测 
 * @param[in] frame_count   滑动窗口里图像个数，即当前图像帧在滑窗中的索引
 * 
 * @return double 特征点在上一次和上上次观测之间的视差 = srqt(du * du + dv * dv)
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count) {
  const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];  // 获得 当前帧特征点 在上上次观测到的情况
  const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];  // 获得 当前帧特征点 在上次观测到的情况

  double ans = 0;
  Vector3d p_j = frame_j.point;   // 获得 特征点在上次观测时的 归一化平面坐标

  double u_j = p_j(0);
  double v_j = p_j(1);

  Vector3d p_i = frame_i.point;  // 获得 特征点在上上次观测时的 归一化平面坐标
  Vector3d p_i_comp;

  //int r_i = frame_count - 2;
  //int r_j = frame_count - 1;
  //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
  p_i_comp = p_i;
  // 齐次坐标转非齐次坐标，无意义，因为在定义变量的时候 p_i(2) == 1
  double dep_i = p_i(2);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  double du = u_i - u_j, dv = v_i - v_j;

  // 重复操作无意义
  double dep_i_comp = p_i_comp(2);
  double u_i_comp = p_i_comp(0) / dep_i_comp;
  double v_i_comp = p_i_comp(1) / dep_i_comp;
  double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

  // 等价于 ans = max(ans, sqrt(du * du + dv * dv));
  ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

  return ans;
}