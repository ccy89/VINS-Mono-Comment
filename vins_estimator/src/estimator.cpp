#include "estimator.h"

Estimator::Estimator() : f_manager{Rs} {
  ROS_INFO("init begins");
  clearState();
}

void Estimator::setParameter() {
  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic[i] = TIC[i];
    ric[i] = RIC[i];
  }
  f_manager.setRic(ric);
  ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  td = TD;
}

void Estimator::clearState() {
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    Rs[i].setIdentity();
    Ps[i].setZero();
    Vs[i].setZero();
    Bas[i].setZero();
    Bgs[i].setZero();
    dt_buf[i].clear();
    linear_acceleration_buf[i].clear();
    angular_velocity_buf[i].clear();

    if (pre_integrations[i] != nullptr)
      delete pre_integrations[i];
    pre_integrations[i] = nullptr;
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic[i] = Vector3d::Zero();
    ric[i] = Matrix3d::Identity();
  }

  for (auto &it : all_image_frame) {
    if (it.second.pre_integration != nullptr) {
      delete it.second.pre_integration;
      it.second.pre_integration = nullptr;
    }
  }

  solver_flag = INITIAL;
  first_imu = false,
  sum_of_back = 0;
  sum_of_front = 0;
  frame_count = 0;
  solver_flag = INITIAL;
  initial_timestamp = 0;
  all_image_frame.clear();
  td = TD;

  if (tmp_pre_integration != nullptr)
    delete tmp_pre_integration;
  if (last_marginalization_info != nullptr)
    delete last_marginalization_info;

  tmp_pre_integration = nullptr;
  last_marginalization_info = nullptr;
  last_marginalization_parameter_blocks.clear();

  f_manager.clearState();

  failure_occur = 0;
  relocalization_info = 0;

  drift_correct_r = Matrix3d::Identity();
  drift_correct_t = Vector3d::Zero();
}

/**
 * @brief   IMU中值积分，中值积分得到当前帧PQV作为优化初值 
 * 
 * @param[in] dt                  前后两帧IMU数据时间差
 * @param[in] linear_acceleration 当前帧加速度值
 * @param[in] angular_velocity    当前帧角速度值
 */
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity) {
  if (!first_imu) {
    // 第一帧IMU，
    first_imu = true;
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }

  if (!pre_integrations[frame_count]) {
    // 初始化 预积分数据结构
    pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
  }

  if (frame_count != 0) {
    // 不是第一个图像帧，进行IMU预积分，作为当前帧的初始值

    // 这里的积分是两帧图像之间的增量
    pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);  // 传入IMU数据，进行中值积分
    //if(solver_flag != NON_LINEAR)
    tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);  // 传入IMU数据，进行中值积分

    dt_buf[frame_count].push_back(dt);                                    // 存储 dt
    linear_acceleration_buf[frame_count].push_back(linear_acceleration);  // 存储 加速度值
    angular_velocity_buf[frame_count].push_back(angular_velocity);        // 存储 角速度值

    // 这里的积分是获得当前的位姿
    int j = frame_count;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;                // 计算上一帧 a_w0 = R_wb0 * (a_b0 - b_a) - g
    Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];     // 计算中值   ω_b = 0.5 * (ω_b0 + ω_b1) - b_g
    Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();        // 中值积分项 R_wb1 = R_wb0 * ΔR_b0b1
    Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;  // 计算当前帧 a_w1 = R_wb1 * (a_b1 - b_a) - g
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);                   // 计算中值   a_w = 0.5 * (a_w0 + a_w1)
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;                    // 中值积分项 P_wb1 = P_wb0 + V_w0 * dt + 0.5 * a_w * dt^2
    Vs[j] += dt * un_acc;                                            // 中值积分项 V_w1 = V_w0 + a_w * dt
  }

  // 当前帧变上一帧
  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}

/**
 * @brief 处理最新帧图像
 * Step 1: 判断次新帧是否为关键帧，决定边缘化方式
 * Step 2: 如果配置文件没有提供IMU和相机之间的外参，则标定该参数
 * Step 3: 如果系统还未初始化，则初始化系统
 * Step 4: 三角化求解特征点在当前帧的深度、滑窗后端优化，执行边缘化操作，计算先验信息
 * Step 5: 判断系统是否出错，一旦检测到故障，系统将切换回初始化阶段
 * Step 6: 执行滑动窗口，丢弃边缘化帧的观测
 * Step 7: 删除优化后的特征外点
 * 
 * @param[in] image   当前图像特征跟踪情况，数据结构为 <特征点id，<相机id，<特征点归一化坐标，特征点像素坐标，特征点像素速度>>>
 * @param[in] header  当前帧图像时间戳
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header) {
  ROS_DEBUG("new image coming ------------------------------------------");
  ROS_DEBUG("Adding feature points %lu", image.size());

  // Step 1: 判断次新帧是否为关键帧，决定边缘化方式
  // 将当前帧图像 检测到的特征点 添加到 feature容器中，计算每一个点跟踪的次数，以及它的视差
  // 通过检测 上一帧和上上帧图像之间的视差 | 当前图像为是否滑动窗口起始帧2帧 来决定上一帧是否作为关键帧
  if (f_manager.addFeatureCheckParallax(frame_count, image, td)) {
    // 上一帧是关键帧，则后端优化时移除滑窗中的第一帧，当前帧插入滑窗末尾
    marginalization_flag = MARGIN_OLD;
  } else {
    // 上一帧不是关键帧，则后端优化时直接移除滑动窗口中的最后一帧，当前帧插入滑窗末尾
    marginalization_flag = MARGIN_SECOND_NEW;
  }

  ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
  ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
  ROS_DEBUG("Solving %d", frame_count);
  ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());

  Headers[frame_count] = header;  // 记录当前帧图像的时间戳

  ImageFrame imageframe(image, header.stamp.toSec());                                           // 构造图像帧
  imageframe.pre_integration = tmp_pre_integration;                                             // 存储 预积分量
  all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));                          // 存储当前图像帧
  tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};  // 初始化新的 预积分量

  // Step 2: 如果配置文件没有提供IMU和相机之间的外参，则标定该参数，只标定R_bc，和IMU一体的相机，t_bc很小，可以直接忽略
  if (ESTIMATE_EXTRINSIC == 2) {
    ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    if (frame_count != 0) {
      // 找到当前帧和上一帧共视的特征点
      vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
      
      Matrix3d calib_ric;
      if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric)) {
        ROS_WARN("initial extrinsic rotation calib success");
        ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                       << calib_ric);
        ric[0] = calib_ric;
        RIC[0] = calib_ric;
        ESTIMATE_EXTRINSIC = 1;
      }
    }
  }

  if (solver_flag == INITIAL) {        
    // Step 3: 如果系统还未初始化，则初始化系统
    if (frame_count == WINDOW_SIZE) {  // 直到滑动窗口满了，确保有足够的图像帧参与初始化
      bool result = false;
      if (ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1) {  //有外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
        result = initialStructure();                // 视觉惯性联合初始化
        initial_timestamp = header.stamp.toSec();   // 更新初始化时间戳
      }

      if (result) {  // 初始化成功
        // 先进行一次滑动窗口非线性优化，得到当前帧与第一帧的位姿
        solver_flag = NON_LINEAR;
        
        // 再次三角化特征点后，执行后端优化
        // 这里三角化特征点是多余的，因为在初始化中已经进行三角化了
        solveOdometry();

        slideWindow();               // 滑动窗口
        f_manager.removeFailures();  // 移除优化后的外点
        ROS_INFO("Initialization finish!");
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];

      } else {  
        slideWindow();  // 初始化失败则直接滑动窗口
      }
    } else {
      ++frame_count;  // 滑动窗口没有满，图像帧数量+1
    }
  } else {  
    // 初始化结束
    TicToc t_solve;
    // Step 4: 三角化求解特征点在当前帧的深度、滑窗后端优化，执行边缘化操作，计算先验信息
    solveOdometry();
    ROS_DEBUG("solver costs: %fms", t_solve.toc());

    // Step 5: 判断系统是否出错，一旦检测到故障，系统将切换回初始化阶段
    if (failureDetection()) {  
      ROS_WARN("failure detection!");
      failure_occur = 1;
      clearState();
      setParameter();
      ROS_WARN("system reboot!");
      return;
    }

    TicToc t_margin;

    // Step 6: 执行滑动窗口，丢弃边缘化帧的观测
    slideWindow();

    // Step 7: 删除优化后的特征外点
    f_manager.removeFailures();
    ROS_DEBUG("marginalization costs: %fms", t_margin.toc());

    // 存储当前滑窗中的相机位姿，用于发布
    key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
      key_poses.push_back(Ps[i]);

    // 当前帧变成上一帧
    last_R = Rs[WINDOW_SIZE];
    last_P = Ps[WINDOW_SIZE];
    last_R0 = Rs[0];
    last_P0 = Ps[0];
  }
}

/**
 * @brief 视觉惯性联合优化
 * Step 1: 通过加速度标准差判断IMU是否有充分运动以初始化
 * Step 2: 将当前滑窗中的所有特征保存到存有 SFMFeature 对象的 sfm_f 中
 * Step 3: relativePose()找到滑窗内和最新帧具有足够视差的图像帧，计算F矩阵并分解得到R、t作为初始值
 * Step 4: sfm.construct() 全局纯视觉SFM，优化滑动窗口帧的位姿
 * Step 5: 利用SFM的结果，更新或计算至今所有的图像帧的位姿
 * Step 6: visualInitialAlign() 将视觉信息和IMU预积分进行对齐
 * 
 * @return true   初始化成功
 * @return false  初始化失败
 */
bool Estimator::initialStructure() {
  TicToc t_sfm;
  // 通过加速度标准差判断IMU是否有充分运动以初始化。
  {
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;             // 获得 当前图像帧预积分量的 dt
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;  // 获得 加速度
      sum_g += tmp_g;                                                   // 累加平均加速度
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);  // 获得 滑窗内的平均加速度

    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);  // 方差
      //cout << "frame g " << tmp_g.transpose() << endl;
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));  // 标准差

    if (var < 0.25) {
      ROS_INFO("IMU excitation not enouth!");   // IMU 运动不够充分
      //return false;
    }
  }

  /** 全局 sfm，将当前滑窗中的所有特征保存到存有 SFMFeature 对象的 sfm_f 中 **/
  Quaterniond Q[frame_count + 1];  // 
  Vector3d T[frame_count + 1];     // 
  map<int, Vector3d> sfm_tracked_points;
  vector<SFMFeature> sfm_f;

  // 遍历特征，将特征存入 sfm_f中，用于初始化
  for (auto &it_per_id : f_manager.feature) {
    int imu_j = it_per_id.start_frame - 1;
    SFMFeature tmp_feature;
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;  // 获得特征id
    // 遍历当前特征的观测
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      Vector3d pts_j = it_per_frame.point;
      tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  }

  // 第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
  // 此处的 relative_R，relative_T 为 当前帧 到 参考帧（第l帧）的坐标系变换Rt
  Matrix3d relative_R;  // R_cl
  Vector3d relative_T;  // t_cl
  int l;

  // 保证具有足够的视差，由F矩阵恢复Rt
  if (!relativePose(relative_R, relative_T, l)) {
    ROS_INFO("Not enough features or parallax; Move device around");
    return false;
  }

  // 对窗口中每个图像帧求解sfm问题，获得如下结果:
  // Q[] 滑窗每一帧的旋转 R_lci
  // T[] 滑窗每一帧的旋转t_lci
  // sfm_tracked_points[] 特征点的世界坐标 P_l
  GlobalSFM sfm;
  if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points)) {
    ROS_DEBUG("global SFM failed!");
    marginalization_flag = MARGIN_OLD;  // 初始化失败，边缘化最老帧
    return false;
  }
  /******************* 至此初始化的存纯视觉部分完成 *****************/

  // 对于所有的图像帧，包括滑窗外的，计算每一帧的位姿
  // 滑窗内的图像帧: 更新后为sfm求解后的值
  // 滑窗外的图像帧: 关联滑窗内的图像帧，使用 solvePnP() 求解位姿
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {    // 遍历 所有图像帧
    cv::Mat r, rvec, t, D, tmp_r;
    if ((frame_it->first) == Headers[i].stamp.toSec()) {    
      // 利用时间戳确定，如果当前图像帧为滑窗内的图像帧，设置为关键帧，并且赋值为BA优化后的值
      frame_it->second.is_key_frame = true;   // 设置为关键帧
      // 将
      frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();    // 转换到 IMU坐标系下 R_lb
      frame_it->second.T = T[i];    // t_lc
      i++;
      continue;
    }

    if ((frame_it->first) > Headers[i].stamp.toSec()) {   // 更新最邻近的关键帧索引
      i++;
    }

    // 将最近的关键帧作为初值，计算当前图像帧的 T_cl
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();  
    Vector3d P_inital = -R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);     // 罗德里格斯公式将旋转矩阵转换成旋转向量
    cv::eigen2cv(P_inital, t);

    frame_it->second.is_key_frame = false;  // 非滑窗内的图像帧不是关键帧
   
    // 当前图像帧能够观测到滑窗内的特征
    vector<cv::Point3f> pts_3_vector;       // 求解 PnP的3d点 P_l
    vector<cv::Point2f> pts_2_vector;       // 求解 PnP的2d点，相机归一化坐标下的前2维信息

    // 遍历当前图像帧观测到的所有特征点
    for (auto &id_pts : frame_it->second.points) {
      int feature_id = id_pts.first;    
      for (auto &i_p : id_pts.second) {
        it = sfm_tracked_points.find(feature_id);   // 寻找当前特征是否出现在滑窗内
        if (it != sfm_tracked_points.end()) {
          // 当前特征出现在滑窗内
          Vector3d world_pts = it->second;    // 获得 特征的世界坐标 P_l
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));  // Eigen 转 opencv
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();    // 获得 特征的归一化坐标的前两维 
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }

    // 所有图像帧都必须满足：至少能够观测到滑窗中5个点，否则不进行初始化
    if (pts_3_vector.size() < 6) {
      cout << "pts_3_vector size " << pts_3_vector.size() << endl;
      ROS_DEBUG("Not enough points for solve pnp !");
      return false;
    }
    
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);    // 由于 用的2d坐标是归一化平面的点，所以相机内参为单位阵

    // PnP 求解 当前图像帧位姿
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
      ROS_DEBUG("solve pnp fail!");
      return false;
    }
    
    cv::Rodrigues(rvec, r);     // 罗德里格斯公式将 旋转向量 转换成 旋转矩阵
    MatrixXd R_pnp, tmp_R_pnp;  // R_cl
    cv::cv2eigen(r, tmp_R_pnp); // opencv -> Eigen 
    R_pnp = tmp_R_pnp.transpose();  // R_lc    

    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);     // t_cl
    T_pnp = R_pnp * (-T_pnp);   // t_lc
    
    // 这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
    frame_it->second.R = R_pnp * RIC[0].transpose();    // R_lb
    frame_it->second.T = T_pnp;   // t_lc
  } // 
  /********************* 求解所有图像帧的位姿结束 ************************/

  // 将 视觉信息 和 IMU预积分 进行对齐
  if (visualInitialAlign())
    return true;
  else {
    ROS_INFO("misalign visual structure with IMU");
    return false;
  }
}


/**
 * @brief 将 视觉轨迹 和 IMU预积分 进行对齐
 * Step 1: 计算陀螺仪偏置，尺度，重力加速度和速度
 * Step 2: 
 *        计算尺度信息、计算初始时刻的位姿、更新所有图像帧在世界坐标系下的位姿
 * 
 * 
 * @return true     对齐成功
 * @return false    对齐失败
 */
bool Estimator::visualInitialAlign() {
  TicToc t_g;
  VectorXd x;
  // 计算陀螺仪偏置，尺度，重力加速度和速度
  bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
  if (!result) {
    ROS_DEBUG("solve g failed!");
    return false;
  }

  // 得到滑窗所有图像帧的位姿Ps、Rs，并将其置为关键帧
  for (int i = 0; i <= frame_count; i++) {
    Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
    Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
    Ps[i] = Pi;
    Rs[i] = Ri;
    all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
  }

  // 将所有特征点的深度置为-1
  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < dep.size(); i++)
    dep[i] = -1;
  f_manager.clearDepth(dep);

  // 重新计算特征点的深度（尺度未校准）, no tic
  Vector3d TIC_TMP[NUM_OF_CAM];
  for (int i = 0; i < NUM_OF_CAM; i++)
    TIC_TMP[i].setZero();
  ric[0] = RIC[0];
  f_manager.setRic(ric);
  f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));


  double s = (x.tail<1>())(0);  // 获得尺度

  // 陀螺仪的偏置bgs改变，重新计算预积分
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }

  // 根据尺度信息s，将 Ps 进行缩放
  for (int i = frame_count; i >= 0; i--) {
    // Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换
    Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
  }
  
  // 根据尺度信息s，将 Vs 进行缩放
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      // Vs为优化得到的速度
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }

  // 根据尺度信息s，将 特征点深度 进行缩放
  for (auto &it_per_id : f_manager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth *= s;
  }

  // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
  Matrix3d R0 = Utility::g2R(g);    // 计算 R_wc0
  // 下面的代码是去掉偏航角，应该是多余的，因为在 g2R()函数里已经做过了
  double yaw = Utility::R2ypr(R0 * Rs[0]).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;


  g = R0 * g;   // 获得世界坐标系下的重力
  //Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;

  // 所有变量从参考坐标系c0旋转到世界坐标系w
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = rot_diff * Ps[i];
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }
  ROS_DEBUG_STREAM("g0     " << g.transpose());
  ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

  return true;
}

/**
 * @brief  判断两帧有足够视差30，且内点数目大于12，则可进行初始化，计算F矩阵，分解得到R和T
 * 
 * @param[out] relative_R   当前帧到第l帧之间的旋转矩阵R
 * @param[out] relative_T   当前帧到第l帧之间的平移向量t
 * @param[out] l            保存滑动窗口中与当前帧满足初始化条件的那一帧
 * 
 * @return true   可以进行初始化
 * @return false  不满足初始化条件
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l) {
  // 遍历滑窗，寻找第i帧到窗口最后一帧（当前帧）的对应特征点
  for (int i = 0; i < WINDOW_SIZE; i++) {
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager.getCorresponding(i, WINDOW_SIZE);  // 获得 第i帧到窗口最后一帧的共视特征点对
    if (corres.size() > 20) {                             // 共视特征点要求大于20

      double sum_parallax = 0;  // 两图像帧总视差
      double average_parallax;  // 两图像平均视差
      // 遍历 共视特征
      for (int j = 0; j < int(corres.size()); j++) {
        Vector2d pts_0(corres[j].first(0), corres[j].first(1));    // 获得 归一化坐标的前两维
        Vector2d pts_1(corres[j].second(0), corres[j].second(1));  // 获得 归一化坐标的前两维
        double parallax = (pts_0 - pts_1).norm();                  // 视差
        sum_parallax = sum_parallax + parallax;                    // 累加视差
      }
      average_parallax = 1.0 * sum_parallax / int(corres.size());  // 平均视差

      // 判断是否满足初始化条件：视差>30 && 内点数满足要求
      // 同时返回窗口最后一帧（当前帧）到第i帧（参考帧）的Rt
      if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)) {
        l = i;  // 找到参考帧了，可以开始初始化
        ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
        return true;
      }
    }
  }
  return false;
}

void Estimator::solveOdometry() {
  if (frame_count < WINDOW_SIZE)
    return;
  if (solver_flag == NON_LINEAR) {
    TicToc t_tri;
    f_manager.triangulate(Ps, tic, ric);    // 三角化特征点
    ROS_DEBUG("triangulation costs %f", t_tri.toc());
    optimization();   // 后端优化
  }
}

/**
 * @brief 由于 ceres 使用数值，因此要将 数组vector 转换成 double数组
 *        Ps、Rs 转变成 para_Pose
 *        Vs、Bas、Bgs 转变成 para_SpeedBias
 *        R_bc 转变为 para_Ex_Pose
 *        逆深度 转变为 para_Feature
 *        td 转变为 para_Td
 */
void Estimator::vector2double() {
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    para_Pose[i][0] = Ps[i].x();
    para_Pose[i][1] = Ps[i].y();
    para_Pose[i][2] = Ps[i].z();
    Quaterniond q{Rs[i]};
    para_Pose[i][3] = q.x();
    para_Pose[i][4] = q.y();
    para_Pose[i][5] = q.z();
    para_Pose[i][6] = q.w();

    para_SpeedBias[i][0] = Vs[i].x();
    para_SpeedBias[i][1] = Vs[i].y();
    para_SpeedBias[i][2] = Vs[i].z();

    para_SpeedBias[i][3] = Bas[i].x();
    para_SpeedBias[i][4] = Bas[i].y();
    para_SpeedBias[i][5] = Bas[i].z();

    para_SpeedBias[i][6] = Bgs[i].x();
    para_SpeedBias[i][7] = Bgs[i].y();
    para_SpeedBias[i][8] = Bgs[i].z();
  }
  for (int i = 0; i < NUM_OF_CAM; i++) {
    para_Ex_Pose[i][0] = tic[i].x();
    para_Ex_Pose[i][1] = tic[i].y();
    para_Ex_Pose[i][2] = tic[i].z();
    Quaterniond q{ric[i]};
    para_Ex_Pose[i][3] = q.x();
    para_Ex_Pose[i][4] = q.y();
    para_Ex_Pose[i][5] = q.z();
    para_Ex_Pose[i][6] = q.w();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    para_Feature[i][0] = dep(i);
  if (ESTIMATE_TD)
    para_Td[0][0] = td;
}


/**
 * @brief 从ceres的结果中恢复优化变量，并且保证滑窗中第一帧相机的偏航角不变
 */
void Estimator::double2vector() {
  // Step 1: 记录 滑窗中第一帧位姿优化前的值
  Vector3d origin_R0 = Utility::R2ypr(Rs[0]);   // 记录优化前滑窗第一帧的偏航角
  Vector3d origin_P0 = Ps[0];                   // 记录优化前滑窗第一帧的 t_wb

  if (failure_occur) {
    origin_R0 = Utility::R2ypr(last_R0);
    origin_P0 = last_P0;
    failure_occur = 0;
  }
  // Step 2: 计算 滑窗中第一帧相机位姿在优化前后的偏航角变化量
  Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                   para_Pose[0][3],
                                                   para_Pose[0][4],
                                                   para_Pose[0][5])
                                           .toRotationMatrix());
  double y_diff = origin_R0.x() - origin_R00.x();   // 求解优化前和优化后，偏航角的变化量
  //TODO
  // 计算偏航角变化量对应的旋转矩阵
  Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
  if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
    ROS_DEBUG("euler singular point!");
    rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                   para_Pose[0][3],
                                   para_Pose[0][4],
                                   para_Pose[0][5])
                           .toRotationMatrix()
                           .transpose();
  }

  // Step 3: 从ceres中恢复优化变量，并且保证滑窗中第一帧相机的偏航角不变
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

    Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) +
            origin_P0;

    Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                para_SpeedBias[i][1],
                                para_SpeedBias[i][2]);

    Bas[i] = Vector3d(para_SpeedBias[i][3],
                      para_SpeedBias[i][4],
                      para_SpeedBias[i][5]);

    Bgs[i] = Vector3d(para_SpeedBias[i][6],
                      para_SpeedBias[i][7],
                      para_SpeedBias[i][8]);
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic[i] = Vector3d(para_Ex_Pose[i][0],
                      para_Ex_Pose[i][1],
                      para_Ex_Pose[i][2]);
    ric[i] = Quaterniond(para_Ex_Pose[i][6],
                         para_Ex_Pose[i][3],
                         para_Ex_Pose[i][4],
                         para_Ex_Pose[i][5])
                 .toRotationMatrix();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    dep(i) = para_Feature[i][0];
  f_manager.setDepth(dep);
  if (ESTIMATE_TD)
    td = para_Td[0][0];

  // 利用回环帧纠正当前滑窗内的相机位姿
  if (relocalization_info) {
    Matrix3d relo_r;
    Vector3d relo_t;
    // 校正 闭环帧的位姿 
    relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
    relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                 relo_Pose[1] - para_Pose[0][1],
                                 relo_Pose[2] - para_Pose[0][2]) + origin_P0;

    double drift_correct_yaw;
    drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();   // 
    drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
    drift_correct_t = prev_relo_t - drift_correct_r * relo_t;

    // 计算 优化后的闭环帧和对应关键帧之间的相对位姿 T_loop_c
    relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);   // t_loop_c = R_w_loop^T * (t_w_c - t_w_loop)
    relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];              // R_loop_c = R_w_loop^T * R_w_c
    relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());   // 只考虑偏航角
    relocalization_info = 0;
  }
}

/**
 * @brief 检测系统是否发生错误
 */
bool Estimator::failureDetection() {
  if (f_manager.last_track_num < 2) {
    ROS_INFO(" little feature %d", f_manager.last_track_num);
    //return true;
  }
  if (Bas[WINDOW_SIZE].norm() > 2.5) {
    ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
    return true;
  }
  if (Bgs[WINDOW_SIZE].norm() > 1.0) {
    ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
    return true;
  }
  /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
  Vector3d tmp_P = Ps[WINDOW_SIZE];
  if ((tmp_P - last_P).norm() > 5) {
    ROS_INFO(" big translation");
    return true;
  }
  if (abs(tmp_P.z() - last_P.z()) > 1) {
    ROS_INFO(" big z translation");
    return true;
  }
  Matrix3d tmp_R = Rs[WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * last_R;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) {
    ROS_INFO(" big delta_angle ");
    //return true;
  }
  return false;
}


/**
 * @brief 计算后端优化、边缘化先验信息
 *        添加要优化的变量 (p, v, q, ba, bg) 一共15个自由度，IMU 的外参 R_bc 也可以加进来
 *        添加残差，残差项分为4块 先验残差 + IMU残差 + 视觉残差 + 闭环检测残差
 *        根据倒数第二帧是不是关键帧确定边缘化的策略     
 */
void Estimator::optimization() {
  /****************************后端优化*****************************/

  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  // loss_function = new ceres::HuberLoss(1.0);
  loss_function = new ceres::CauchyLoss(1.0);

  // 遍历滑窗，将相机位姿、速度、角速度偏置、加速度偏置添加进优化
  // 优化变量一共15维
  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
    problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
  }

  // 将相机和IMU之间的外参添加进优化问题
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
    if (!ESTIMATE_EXTRINSIC) {
      ROS_DEBUG("fix extinsic param");
      problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    } else
      ROS_DEBUG("estimate extinsic param");
  }

  // 将 IMU和相机之间的时间同步差添加进优化
  if (ESTIMATE_TD) {
    problem.AddParameterBlock(para_Td[0], 1);
    //problem.SetParameterBlockConstant(para_Td[0]);
  }

  TicToc t_whole, t_prepare;
  vector2double();    // 将优化变量转为数组的形式

  // 将 滑窗先验残差添加进优化
  if (last_marginalization_info) {
    // construct new marginlization_factor
    MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
    problem.AddResidualBlock(marginalization_factor, NULL,
                             last_marginalization_parameter_blocks);
  }

  // 添加 IMU 残差项，从 pre_integrations[1] 到 pre_integrations[10]
  for (int i = 0; i < WINDOW_SIZE; i++) {
    int j = i + 1;
    if (pre_integrations[j]->sum_dt > 10.0)
      continue;
    IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
    problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
  }

  int f_m_cnt = 0;
  int feature_index = -1;
  // 添加 视觉重投影残差项
  for (auto &it_per_id : f_manager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    ++feature_index;

    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      if (imu_i == imu_j) {
        continue;
      }
      Vector3d pts_j = it_per_frame.point;
      if (ESTIMATE_TD) {  // 含 td的 重投影误差
        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
        
        problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
      } else {
        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
        problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
      }
      f_m_cnt++;
    }
  }

  ROS_DEBUG("visual measurement count: %d", f_m_cnt);
  ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

  // 如果回环检测线程检测到了闭环，将闭环信息添加进优化
  // 需要开启在配置文件中开启 fast_relocalization = 1
  if (relocalization_info) {
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);    // 添加 闭环帧的位姿
    int retrive_feature_index = 0;    // 共视特征在 match_points 中的索引
    int feature_index = -1;

    // 遍历当前滑窗内的特征，找到滑窗和闭环帧的共视
    for (auto &it_per_id : f_manager.feature) {
      it_per_id.used_num = it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;
      
      ++feature_index;
      int start = it_per_id.start_frame;  // 滑窗内观测到当前特征的第一帧
      if (start <= relo_frame_local_index) {
        while ((int)match_points[retrive_feature_index].z() < it_per_id.feature_id) {
          retrive_feature_index++;
        }
        if ((int)match_points[retrive_feature_index].z() == it_per_id.feature_id) {
          // 闭环帧和对应关键帧的共视特征在滑窗内也有，则添加闭环信息边
          Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0); // 特征在闭环帧的归一化坐标
          Vector3d pts_i = it_per_id.feature_per_frame[0].point;    // 特征在滑窗内第一次观测时的归一化坐标                                                      

          // 这里应该有问题，pts_j是特征在闭环帧的归一化坐标，关联的变量却是 relo_Pose 
          ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);   // 添加闭环帧的重投影误差
          problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
          retrive_feature_index++;
        }
      }
    }
  }

  ceres::Solver::Options options;

  options.linear_solver_type = ceres::DENSE_SCHUR;      // 使用稠密矩阵
  //options.num_threads = 2;  
  options.trust_region_strategy_type = ceres::DOGLEG;   // 使用 dogleg算法
  options.max_num_iterations = NUM_ITERATIONS;          // 设置最大迭代次数
  //options.use_explicit_schur_complement = true;
  //options.minimizer_progress_to_stdout = true;
  //options.use_nonmonotonic_steps = true;
  
  // 设这 优化最大时间 
  if (marginalization_flag == MARGIN_OLD) {
    options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
  } else {
    options.max_solver_time_in_seconds = SOLVER_TIME;
  }
    
  TicToc t_solver;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //cout << summary.BriefReport() << endl;
  ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
  ROS_DEBUG("solver costs: %f", t_solver.toc());

  double2vector();    // 从优化结果中还原出变量

  /************************************边缘化处理***************************************/
  TicToc t_whole_marginalization;
  if (marginalization_flag == MARGIN_OLD) {   // 如果上一帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验
    MarginalizationInfo *marginalization_info = new MarginalizationInfo();
    vector2double();

    // 1、将上一次先验残差项传递给marginalization_info
    if (last_marginalization_info) {
      vector<int> drop_set;
      for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
        if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
            last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
          drop_set.push_back(i);
      }
      
      MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                     last_marginalization_parameter_blocks,
                                                                     drop_set);

      marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    // 2. 将第1帧和第2帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
    {
      if (pre_integrations[1]->sum_dt < 10.0) {
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                       vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                       vector<int>{0, 1});
                      
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    // 3. 最后将第一次观测为滑窗中第1帧的路标点 以及 滑窗中和第1帧共视该路标点的相机添加进marginalization_info中
    {
      int feature_index = -1;
      // 遍历滑窗内所有跟踪到的特征
      for (auto &it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))  // 跳过不存在共视的特征点
          continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        if (imu_i != 0)   // 如果当前特征第一个观察帧不是第1帧就不进行考虑
          continue;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;  // 获得当前特征在第1帧相机归一化平面的坐标
        // 遍历滑窗内观测到当前特征的每一帧
        for (auto &it_per_frame : it_per_id.feature_per_frame) {
          imu_j++;
          if (imu_i == imu_j)
            continue;

          Vector3d pts_j = it_per_frame.point;    // 获得 当前特征 在 相机j 归一化平面的坐标
          if (ESTIMATE_TD) {
            ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                              it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                              it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                           vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          } else {
            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                           vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
        }
      }
    }

    TicToc t_pre_margin;
    // 4、计算每个残差块对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
    marginalization_info->preMarginalize();
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

    TicToc t_margin;
    // 5、执行边缘化：多线程构造先验项舒尔补AX=b的结构，计算舒尔补
    marginalization_info->marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.toc());

    // 6、调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
    std::unordered_map<long, double *> addr_shift;
    // 丢弃掉最老帧
    for (int i = 1; i <= WINDOW_SIZE; i++) {
      addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
      addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
      addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
    if (ESTIMATE_TD) {
      addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
    }
    vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

    if (last_marginalization_info)
      delete last_marginalization_info;
    last_marginalization_info = marginalization_info;           // 记录当前先验信息
    last_marginalization_parameter_blocks = parameter_blocks;   // 记录当前先验信息中非边缘化变量的地址

  } else {    
    // 如果上一帧不是关键帧，边缘化掉次新帧
    // 存在先验边缘化信息时才进行次新帧边缘化；否则仅仅通过slidewindow 丢弃次新帧
    if (last_marginalization_info &&
        std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1])) {
      MarginalizationInfo *marginalization_info = new MarginalizationInfo();    // 构造新的 边缘化信息体
      vector2double();

      // 设置从上一次的先验信息中边缘化次新帧的位姿信息
      if (last_marginalization_info) {
        vector<int> drop_set;   // 记录需要丢弃的变量在last_marginalization_parameter_blocks中的索引
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
          ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
          if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
            drop_set.push_back(i);    // 次新帧在先验信息中的索引
        }

        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);   // 使用上一次先验残差 构建 当前的先验残差
        // 从上一次的先验残差中设置边缘化次新帧的位姿信息
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                       last_marginalization_parameter_blocks,
                                                                       drop_set);

        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      TicToc t_pre_margin;
      ROS_DEBUG("begin marginalization");
      // 2、边缘化预处理：遍历所有残差块，计算Jacobian矩阵 和 残差(residuals)
      marginalization_info->preMarginalize();
      ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

      TicToc t_margin;
      ROS_DEBUG("begin marginalization");
      // 3、执行边缘化：多线程构建Hessian矩阵，计算舒尔补
      marginalization_info->marginalize();
      ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

      // 4、调整参数块在下一次窗口中对应的位置（去掉次新帧）
      std::unordered_map<long, double *> addr_shift;
      for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (i == WINDOW_SIZE - 1)   // 上一帧被丢弃
          continue;
        else if (i == WINDOW_SIZE) {  // 当前帧覆盖上一帧
          addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
          addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        } else {    // 其他帧不动
          addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
          addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
        }
      }

      for (int i = 0; i < NUM_OF_CAM; i++) {
        addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
      }

      if (ESTIMATE_TD) {
        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
      }

      vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
      if (last_marginalization_info)
        delete last_marginalization_info;
      last_marginalization_info = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;
    }
  }
  ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

  ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

/**
 * @brief 实现滑动窗口
 * 如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息
 * 如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性
 */
void Estimator::slideWindow() {
  TicToc t_margin;
  if (marginalization_flag == MARGIN_OLD) {   // 上一帧是关键帧，则滑走最老的一帧
    // 备份最老帧的时间戳和位姿信息
    double t_0 = Headers[0].stamp.toSec();
    back_R0 = Rs[0];
    back_P0 = Ps[0];
    if (frame_count == WINDOW_SIZE) {
      // 将前后帧数据交换，最终结果为 1 2 3 4 5 6 7 8 9 10 0
      for (int i = 0; i < WINDOW_SIZE; i++) {
        Rs[i].swap(Rs[i + 1]);

        std::swap(pre_integrations[i], pre_integrations[i + 1]);

        dt_buf[i].swap(dt_buf[i + 1]);
        linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
        angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

        Headers[i] = Headers[i + 1];
        Ps[i].swap(Ps[i + 1]);
        Vs[i].swap(Vs[i + 1]);
        Bas[i].swap(Bas[i + 1]);
        Bgs[i].swap(Bgs[i + 1]);
      }

      // 下边这一步的结果应该是 1 2 3 4 5 6 7 8 9 10 10
      Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
      Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
      Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
      Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
      Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
      Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

      // 由于当前帧已经赋值给上一帧了，删除当前帧的信息
      delete pre_integrations[WINDOW_SIZE];
      pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};  // 初始化新的IMU预积分量

      dt_buf[WINDOW_SIZE].clear();
      linear_acceleration_buf[WINDOW_SIZE].clear();
      angular_velocity_buf[WINDOW_SIZE].clear();


      if (true || solver_flag == INITIAL) {
        map<double, ImageFrame>::iterator it_0;
        it_0 = all_image_frame.find(t_0);   // 在 all_image_frame 找到边缘化的最老帧
        // 从 all_image_frame 里面删除 边缘化的最老帧
        delete it_0->second.pre_integration;    
        it_0->second.pre_integration = nullptr;

        for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it) {
          if (it->second.pre_integration)
            delete it->second.pre_integration;
          it->second.pre_integration = NULL;
        }

        all_image_frame.erase(all_image_frame.begin(), it_0);
        all_image_frame.erase(t_0);
      }
      slideWindowOld();   // 处理特征观测信息
    }
  } else {    // 上一帧不是关键帧，则边缘化次新帧
    if (frame_count == WINDOW_SIZE) {
      // 遍历当前帧的IMU数据，将其拼接到上一帧的 IMU预积分量上
      for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
        double tmp_dt = dt_buf[frame_count][i];
        Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
        Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

        pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

        dt_buf[frame_count - 1].push_back(tmp_dt);
        linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
        angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
      }

      // 当前帧变成上一帧，结果为 0 1 2 3 4 5 6 7 8 10 10
      Headers[frame_count - 1] = Headers[frame_count];
      Ps[frame_count - 1] = Ps[frame_count];
      Vs[frame_count - 1] = Vs[frame_count];
      Rs[frame_count - 1] = Rs[frame_count];
      Bas[frame_count - 1] = Bas[frame_count];
      Bgs[frame_count - 1] = Bgs[frame_count];

      // 由于当前帧已经赋值给上一帧了，删除当前帧的信息
      delete pre_integrations[WINDOW_SIZE];
      pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};  // 初始化新的IMU预积分量

      dt_buf[WINDOW_SIZE].clear();
      linear_acceleration_buf[WINDOW_SIZE].clear();
      angular_velocity_buf[WINDOW_SIZE].clear();

      slideWindowNew();   // 处理特征观测信息
    }
  }
}

/**
 * @brief 滑出次新帧，删除次新帧对特征点的观测
 */
void Estimator::slideWindowNew() {
  sum_of_front++;
  f_manager.removeFront(frame_count);   // 删除次新帧对特征的观测
}

/**
 * @brief 滑出最老帧，删除最老帧对特征点的观测
 */
void Estimator::slideWindowOld() {
  sum_of_back++;

  bool shift_depth = solver_flag == NON_LINEAR ? true : false;
  if (shift_depth) {
    // back_R0、back_P0为窗口中最老帧的位姿
    // Rsp[0]、Ps[0] 为当前滑动窗口后第1帧的位姿，即原来的第2帧
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_R0 * ric[0];
    R1 = Rs[0] * ric[0];
    P0 = back_P0 + back_R0 * tic[0];
    P1 = Ps[0] + Rs[0] * tic[0];
    f_manager.removeBackShiftDepth(R0, P0, R1, P1);   // 特征点删除最老帧的观测，将观测传递到滑窗新的第一帧（原来的第二帧）
  } else
    f_manager.removeBack();   // 特征点直接删除最老帧的观测
}


/**
 * @brief 设置闭环帧的信息
 * 
 * @param[in] _frame_stamp    闭环帧对应的关键帧的时间戳
 * @param[in] _frame_index    闭环帧的ID
 * @param[in] _match_points   闭环帧和对应关键帧共视的特征点，前2维是特征点在闭环帧的归一化坐标，第3维是特征ID
 * @param[in] _relo_t         闭环帧位姿 R_wb
 * @param[in] _relo_r         闭环帧位姿 t_wb
 */
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r) {
  relo_frame_stamp = _frame_stamp;
  relo_frame_index = _frame_index;
  match_points.clear();

  // 闭环帧和对应的关键帧共视的特征点
  match_points = _match_points;

  // 记录闭环帧的位姿
  prev_relo_t = _relo_t;          
  prev_relo_r = _relo_r;
  
  
  for (int i = 0; i < WINDOW_SIZE; i++) {
    if (relo_frame_stamp == Headers[i].stamp.toSec()) {   // 找到闭环帧对应的关键帧在当前滑窗的哪个位置
      relo_frame_local_index = i;   // 闭环帧在关键帧列表中的索引
      relocalization_info = 1;      // 标记在优化中使用闭环帧残差项
      
      // 原代码，relo_Pose 是闭环帧对应的关键帧的位姿
      // for (int j = 0; j < SIZE_POSE; j++)
      //   relo_Pose[j] = para_Pose[i][j];     

      // 修改后 relo_Pose 是闭环帧在前端VIO世界坐标下的位姿
      relo_Pose[0] = prev_relo_t.x();
      relo_Pose[1] = prev_relo_t.y();
      relo_Pose[2] = prev_relo_t.z();

      Eigen::Quaterniond tempR(prev_relo_r);
      relo_Pose[3] = tempR.x();
      relo_Pose[4] = tempR.y();
      relo_Pose[5] = tempR.z();
      relo_Pose[6] = tempR.w();
    }
  }
}
