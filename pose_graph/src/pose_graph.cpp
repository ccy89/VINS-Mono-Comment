#include "pose_graph.h"

PoseGraph::PoseGraph() {
  posegraph_visualization = new CameraPoseVisualization(1.0, 0.0, 1.0, 1.0);
  posegraph_visualization->setScale(0.1);
  posegraph_visualization->setLineWidth(0.01);
  t_optimization = std::thread(&PoseGraph::optimize4DoF, this);   // 开启ceres优化线程
  earliest_loop_index = -1;
  t_drift = Eigen::Vector3d(0, 0, 0);
  yaw_drift = 0;
  r_drift = Eigen::Matrix3d::Identity();
  w_t_vio = Eigen::Vector3d(0, 0, 0);
  w_r_vio = Eigen::Matrix3d::Identity();
  global_index = 0;
  sequence_cnt = 0;
  sequence_loop.push_back(0);   // 地图
  base_sequence = 1;            
}

PoseGraph::~PoseGraph() {
  t_optimization.join();
}

void PoseGraph::registerPub(ros::NodeHandle& n) {
  pub_pg_path = n.advertise<nav_msgs::Path>("pose_graph_path", 1000);
  pub_base_path = n.advertise<nav_msgs::Path>("base_path", 1000);
  pub_pose_graph = n.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);
  for (int i = 1; i < 10; i++)
    pub_path[i] = n.advertise<nav_msgs::Path>("path_" + to_string(i), 1000);
}

/**
 * @brief PoseGraph 加载 BRIEF词典，用于特征匹配
 * 
 * @param[in] voc_path  词典的路径
 */
void PoseGraph::loadVocabulary(std::string voc_path) {
  voc = new BriefVocabulary(voc_path);
  db.setVocabulary(*voc, false, 0);
}

/**
 * @brief 添加关键帧，进行回环检测与闭环
 * 
 * @param[in] cur_kf            新添加的关键帧
 * @param[in] flag_detect_loop  是否进行回环检测
 */
void PoseGraph::addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop) {
  //shift to base frame
  Vector3d vio_P_cur;
  Matrix3d vio_R_cur;

  if (sequence_cnt != cur_kf->sequence) {
    // 如果 关键帧所属的轨迹序列 和 PoseGraph当前所处的轨迹序列不同，则创建新的轨迹序列
    sequence_cnt++;
    sequence_loop.push_back(0);
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    m_drift.lock();
    t_drift = Eigen::Vector3d(0, 0, 0);
    r_drift = Eigen::Matrix3d::Identity();
    m_drift.unlock();
  }

  cur_kf->getVioPose(vio_P_cur, vio_R_cur);     // 获得 当前关键帧在前端VIO中的位姿 Twb
  // 计算 当前关键帧在当前轨迹坐标系下的位姿 T_vio_cur （没有和其他轨迹序列发生过闭环）
  // 计算 当前关键帧在w1坐标系下的位姿 T_w1_cur （和其他轨迹序列发生过闭环）
  vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
  vio_R_cur = w_r_vio * vio_R_cur;            

  cur_kf->updateVioPose(vio_P_cur, vio_R_cur);  // 更新 当前关键帧的位姿
  cur_kf->index = global_index;                 // 标记当前关键帧的ID
  global_index++;
  int loop_index = -1;

  if (flag_detect_loop) {
    // 如果需要进行回环检测，找到闭环帧
    TicToc tmp_t;
    loop_index = detectLoop(cur_kf, cur_kf->index);   // 回环检测，返回闭环候选帧的索引
  } else {
    // 如果不进行回环检测，则将关键帧的Brief描述子信息添加到数据库之中
    addKeyFrameIntoVoc(cur_kf);
  }
  if (loop_index != -1) {
    //printf(" %d detect loop with %d \n", cur_kf->index, loop_index);
    KeyFrame* old_kf = getKeyFrame(loop_index);   // 获取闭环候选帧

    // findConnection() 计算 闭环帧和当前帧之间在当前帧世界坐标系下的相对位姿，T_loop_cur
    // 如果启用 fast_relocalization 功能，findConnection() 会发布闭环帧相关信息
    if (cur_kf->findConnection(old_kf)) {
      if (earliest_loop_index > loop_index || earliest_loop_index == -1)
        earliest_loop_index = loop_index;       // 更新最早的闭环帧

      Vector3d w_P_old, w_P_cur, vio_P_cur;
      Matrix3d w_R_old, w_R_cur, vio_R_cur; 
      old_kf->getVioPose(w_P_old, w_R_old);     // 获得闭环帧在w1坐标系下的位姿 T_w1_loop
      
      // 获得当前帧在自身所在轨迹的 T_vio_cur （没有和其他轨迹发生过闭环）
      // 获得当前关键帧在w1坐标系下的位姿 T_w1_cur（和其他轨迹发生过闭环）
      cur_kf->getVioPose(vio_P_cur, vio_R_cur); 
      
      // 当前关键帧和闭环帧处在不同的轨迹序列，将当前轨迹序列的关键帧都校正到世界坐标系w1之下
      if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0) {
        // 获取 利用PnP计算得到的 当前帧与闭环帧的相对位姿 T_loop_cur
        Vector3d relative_t;
        Quaterniond relative_q;
        relative_t = cur_kf->getLoopRelativeT();                      // t_loop_cur
        relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix(); // R_loop_cur

        // 利用闭环帧位姿和相对位姿，计算当前帧在w1坐标系下的位姿 T_w1_cur
        w_P_cur = w_R_old * relative_t + w_P_old;
        w_R_cur = w_R_old * relative_q;         
        
        // 计算 当前帧和闭环帧轨迹偏移量 T_w1_vio
        double shift_yaw;
        Matrix3d shift_r;
        Vector3d shift_t;
        shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();  // 使用IMU，优化问题是4DoF，只考虑偏航角
        shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));                       
        shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;
        
        // 更新 T_w1_vio
        w_r_vio = shift_r;
        w_t_vio = shift_t;
        // 更新 T_w1_cur
        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
        vio_R_cur = w_r_vio * vio_R_cur; 
        cur_kf->updateVioPose(vio_P_cur, vio_R_cur);    // 更新 当前关键帧的位姿为 T_w1_cur

        list<KeyFrame*>::iterator it = keyframelist.begin();
        // 遍历 所有关键帧，将所有和当前关键帧处在统一轨迹序列的关键帧都拉到w1之下
        for (; it != keyframelist.end(); it++) {
          if ((*it)->sequence == cur_kf->sequence) {  // 和当前关键帧处在统一队列
            Vector3d vio_P_cur;
            Matrix3d vio_R_cur;
            (*it)->getVioPose(vio_P_cur, vio_R_cur);
            vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
            vio_R_cur = w_r_vio * vio_R_cur;
            (*it)->updateVioPose(vio_P_cur, vio_R_cur);
          }
        }
        sequence_loop[cur_kf->sequence] = 1;    // 标记当前轨迹已经有和w1之间的转换矩阵了
      }

      m_optimize_buf.lock();
      optimize_buf.push(cur_kf->index);   // 将当前帧放入优化队列中，进行4DoF闭环优化
      m_optimize_buf.unlock();
    }
  }

  m_keyframelist.lock();
  Vector3d P;
  Matrix3d R;
  cur_kf->getVioPose(P, R);
  P = r_drift * P + t_drift;
  R = r_drift * R;
  cur_kf->updatePose(P, R);
  Quaterniond Q{R};
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time(cur_kf->time_stamp);
  pose_stamped.header.frame_id = "world";
  pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
  pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
  pose_stamped.pose.position.z = P.z();
  pose_stamped.pose.orientation.x = Q.x();
  pose_stamped.pose.orientation.y = Q.y();
  pose_stamped.pose.orientation.z = Q.z();
  pose_stamped.pose.orientation.w = Q.w();
  path[sequence_cnt].poses.push_back(pose_stamped);
  path[sequence_cnt].header = pose_stamped.header;

  // 保存关键帧位姿
  if (SAVE_LOOP_PATH) {
    ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
    loop_path_file.setf(ios::fixed, ios::floatfield);
    loop_path_file.precision(10);
    loop_path_file << cur_kf->time_stamp << " ";
    loop_path_file.precision(5);
    loop_path_file << P.x() << " "
                   << P.y() << " "
                   << P.z() << " "
                   << Q.x() << " "
                   << Q.y() << " "
                   << Q.z() << " "
                   << Q.w() << endl;
    loop_path_file.close();
  }
  //draw local connection
  if (SHOW_S_EDGE) {
    list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin();
    for (int i = 0; i < 4; i++) {
      if (rit == keyframelist.rend())
        break;
      Vector3d conncected_P;
      Matrix3d connected_R;
      if ((*rit)->sequence == cur_kf->sequence) {
        (*rit)->getPose(conncected_P, connected_R);
        posegraph_visualization->add_edge(P, conncected_P);
      }
      rit++;
    }
  }
  if (SHOW_L_EDGE) {
    if (cur_kf->has_loop) {
      //printf("has loop \n");
      KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
      Vector3d connected_P, P0;
      Matrix3d connected_R, R0;
      connected_KF->getPose(connected_P, connected_R);
      //cur_kf->getVioPose(P0, R0);
      cur_kf->getPose(P0, R0);
      if (cur_kf->sequence > 0) {
        //printf("add loop into visual \n");
        posegraph_visualization->add_loopedge(P0, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
      }
    } 
  }
  //posegraph_visualization->add_pose(P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0), Q);

  keyframelist.push_back(cur_kf);
  publish();
  m_keyframelist.unlock();
}

void PoseGraph::loadKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop) {
  cur_kf->index = global_index;
  global_index++;
  int loop_index = -1;
  if (flag_detect_loop)
    loop_index = detectLoop(cur_kf, cur_kf->index);
  else {
    addKeyFrameIntoVoc(cur_kf);
  }
  if (loop_index != -1) {
    printf(" %d detect loop with %d \n", cur_kf->index, loop_index);
    KeyFrame* old_kf = getKeyFrame(loop_index);
    if (cur_kf->findConnection(old_kf)) {
      if (earliest_loop_index > loop_index || earliest_loop_index == -1)
        earliest_loop_index = loop_index;
      m_optimize_buf.lock();
      optimize_buf.push(cur_kf->index);
      m_optimize_buf.unlock();
    }
  }
  m_keyframelist.lock();
  Vector3d P;
  Matrix3d R;
  cur_kf->getPose(P, R);
  Quaterniond Q{R};
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time(cur_kf->time_stamp);
  pose_stamped.header.frame_id = "world";
  pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
  pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
  pose_stamped.pose.position.z = P.z();
  pose_stamped.pose.orientation.x = Q.x();
  pose_stamped.pose.orientation.y = Q.y();
  pose_stamped.pose.orientation.z = Q.z();
  pose_stamped.pose.orientation.w = Q.w();
  base_path.poses.push_back(pose_stamped);
  base_path.header = pose_stamped.header;

  //draw local connection
  if (SHOW_S_EDGE) {
    list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin();
    for (int i = 0; i < 1; i++) {
      if (rit == keyframelist.rend())
        break;
      Vector3d conncected_P;
      Matrix3d connected_R;
      if ((*rit)->sequence == cur_kf->sequence) {
        (*rit)->getPose(conncected_P, connected_R);
        posegraph_visualization->add_edge(P, conncected_P);
      }
      rit++;
    }
  }
  /*
    if (cur_kf->has_loop)
    {
        KeyFrame* connected_KF = getKeyFrame(cur_kf->loop_index);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P,  connected_R);
        posegraph_visualization->add_loopedge(P, connected_P, SHIFT);
    }
    */

  keyframelist.push_back(cur_kf);
  //publish();
  m_keyframelist.unlock();
}


/**
 * @brief 从关键帧序列中获得关键帧
 * 
 * @param[in] index   关键帧的索引
 * @return KeyFrame*  关键帧
 */
KeyFrame* PoseGraph::getKeyFrame(int index) {
  //    unique_lock<mutex> lock(m_keyframelist);
  list<KeyFrame*>::iterator it = keyframelist.begin();
  for (; it != keyframelist.end(); it++) {
    if ((*it)->index == index)
      break;
  }
  if (it != keyframelist.end())
    return *it;
  else
    return NULL;
}


/**
 * @brief 闭环检测
 * 
 * @param[in] keyframe      当前关键帧
 * @param[in] frame_index   当前关键帧在PoseGraph中的索引
 * @return int  闭环候选帧的索引，-1代表没有闭环候选帧
 */
int PoseGraph::detectLoop(KeyFrame* keyframe, int frame_index) {
  // put image into image_pool; for visualization
  cv::Mat compressed_image;
  if (DEBUG_IMAGE) {
    int feature_num = keyframe->keypoints.size();
    cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
    putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
    image_pool[frame_index] = compressed_image;
  }
  TicToc tmp_t;
  //first query; then add this frame into database!
  QueryResults ret;
  TicToc t_query;
  // 查询Brief数据库，得到当前关键帧与数据库中每一个关键帧的相似度评分 ret
  db.query(keyframe->brief_descriptors, ret, 4, frame_index - 50);    
  //printf("query time: %f", t_query.toc());
  //cout << "Searching for Image " << frame_index << ". " << ret << endl;

  TicToc t_add;
  db.add(keyframe->brief_descriptors);    // 添加当前关键帧到字典数据库中
  //printf("add feature time: %f", t_add.toc());
  // ret[0] is the nearest neighbour's score. threshold change with neighour score

  bool find_loop = false;
  cv::Mat loop_result;
  if (DEBUG_IMAGE) {
    loop_result = compressed_image.clone();
    if (ret.size() > 0)
      putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
  }
  // visual loop result
  if (DEBUG_IMAGE) {
    for (unsigned int i = 0; i < ret.size(); i++) {
      int tmp_index = ret[i].Id;
      auto it = image_pool.find(tmp_index);
      cv::Mat tmp_image = (it->second).clone();
      putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
      cv::hconcat(loop_result, tmp_image, loop_result);
    }
  }

  // 确保与相邻帧具有好的相似度评分
  if (ret.size() >= 1 && ret[0].Score > 0.05) {
    for (unsigned int i = 1; i < ret.size(); i++) {
      //if (ret[i].Score > ret[0].Score * 0.3)
      // 评分大于0.015则认为是回环候选帧
      if (ret[i].Score > 0.015) {
        find_loop = true;
        int tmp_index = ret[i].Id;
        if (DEBUG_IMAGE && 0) {
          auto it = image_pool.find(tmp_index);
          cv::Mat tmp_image = (it->second).clone();
          putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
          cv::hconcat(loop_result, tmp_image, loop_result);
        }
      }
    }
  }

  // 对于索引值大于50的关键帧才考虑进行回环
  // 返回评分大于0.015的最早的关键帧索引 min_index
  if (find_loop && frame_index > 50) {
    int min_index = -1;
    for (unsigned int i = 0; i < ret.size(); i++) {
      if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
        min_index = ret[i].Id;
    }
    return min_index;
  } else {
    return -1;
  }
}

/**
 * @brief 将当前关键帧的 Brief描述子信息添加到数据库之中
 * 
 * @param[in] keyframe  关键帧
 */
void PoseGraph::addKeyFrameIntoVoc(KeyFrame* keyframe) {
  // put image into image_pool; for visualization
  cv::Mat compressed_image;
  if (DEBUG_IMAGE) {
    int feature_num = keyframe->keypoints.size();
    cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
    putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
    image_pool[keyframe->index] = compressed_image;
  }

  db.add(keyframe->brief_descriptors);
}

/**
 * @brief 4自由度闭环优化
 */
void PoseGraph::optimize4DoF() {
  while (true) {
    int cur_index = -1;
    int first_looped_index = -1;
    m_optimize_buf.lock();
    while (!optimize_buf.empty()) {
      cur_index = optimize_buf.front();           // 取出最新一个待优化帧作为当前帧
      first_looped_index = earliest_loop_index;   // 获得 最早的闭环帧
      optimize_buf.pop();
    }
    m_optimize_buf.unlock();
    if (cur_index != -1) {
      printf("optimize pose graph \n");
      TicToc tmp_t;
      m_keyframelist.lock();
      KeyFrame* cur_kf = getKeyFrame(cur_index);

      int max_length = cur_index + 1;

      // w^t_i   w^q_i
      double t_array[max_length][3];        // 平移数组，其中存放每个关键帧的 t_wb
      Quaterniond q_array[max_length];      // 旋转数组，其中存放每个关键帧的 R_wb
      double euler_array[max_length][3];
      double sequence_array[max_length];

      ceres::Problem problem;
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      //options.minimizer_progress_to_stdout = true;
      //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      ceres::LossFunction* loss_function;
      loss_function = new ceres::HuberLoss(0.1);
      //loss_function = new ceres::CauchyLoss(1.0);
      ceres::LocalParameterization* angle_local_parameterization =
          AngleLocalParameterization::Create();

      list<KeyFrame*>::iterator it;

      int i = 0;
      // 从最早的闭环帧开始构建优化问题
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < first_looped_index)
          continue;
        (*it)->local_index = i;             // 关键帧在参与优化的关键帧序列中的索引
        Quaterniond tmp_q;
        Matrix3d tmp_r;
        Vector3d tmp_t;
        (*it)->getVioPose(tmp_t, tmp_r);    // 获得关键帧位姿 T_w1_cur
        tmp_q = tmp_r;
        // vector --> array 
        t_array[i][0] = tmp_t(0);
        t_array[i][1] = tmp_t(1);
        t_array[i][2] = tmp_t(2);
        q_array[i] = tmp_q;

        Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());  //  计算欧拉角
        euler_array[i][0] = euler_angle.x();
        euler_array[i][1] = euler_angle.y();
        euler_array[i][2] = euler_angle.z();

        sequence_array[i] = (*it)->sequence;

        problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization); // ceres添加变量 偏航角
        problem.AddParameterBlock(t_array[i], 3);                                   // ceres添加变量 t_w1_cur

        // 固定 当前序列最早的闭环帧 和 地图中载入的关键帧
        if ((*it)->index == first_looped_index || (*it)->sequence == 0) {
          problem.SetParameterBlockConstant(euler_array[i]);
          problem.SetParameterBlockConstant(t_array[i]);
        }

        // 添加序列边，每帧分别与其前边最多四帧构成序列边
        // 这个 CostFuction 在第一次迭代的时候，观测值和估计值是同一个东西
        // 但是由于闭环边的约束，第二次迭代就会不一样
        // 如果只保使用这个 CostFuction，误差始终为0，即不优化，最后回环路径和VIO路径是一样的
        for (int j = 1; j < 5; j++) {
          if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
            // k = i-j

            // 计算 R_wk 对应的欧拉角
            Vector3d euler_conncected = Utility::R2ypr(q_array[i - j].toRotationMatrix());  
            
            // 计算 t_bk_bi = R_w1_bk^T (t_w1_bi - t_w1_bk)
            Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1], t_array[i][2] - t_array[i - j][2]);
            relative_t = q_array[i - j].inverse() * relative_t;

            // 计算 yaw_bk_bi = yaw_bi - yaw_bk (向量方向为指向被减的向量)
            double relative_yaw = euler_array[i][0] - euler_array[i - j][0];  // 计算 R_ki 对应的偏航角

            // 构建 4DoF CostFuction
            // 该 CostFuction 计算在优化过程中，关键帧i、k之间的相对位姿（估计值） 和 开始优化前观测到的相对位姿（观测值） 之间的残差
            ceres::CostFunction* cost_function = FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                      relative_yaw, euler_conncected.y(), euler_conncected.z());
            
            // 这里的 euler_array[i - j] 和 euler_array[i] 只会用到 euler_array[i - j][0] 和 euler_array[i][0]
            problem.AddResidualBlock(cost_function, NULL, euler_array[i - j],
                                     t_array[i - j],
                                     euler_array[i],
                                     t_array[i]);
          }
        }

        // 添加闭环边，将所有过往检测到闭环的两帧，构建误差
        // 这个是主要优化
        if ((*it)->has_loop) {
          assert((*it)->loop_index >= first_looped_index);

          int connected_index = getKeyFrame((*it)->loop_index)->local_index;    // 获得 关键帧对应的闭环帧的索引

          Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());  // 计算闭环帧 R_w1_loop 对应的 欧拉角
          Vector3d relative_t;
          relative_t = (*it)->getLoopRelativeT();               // 获得关键帧和闭环帧之间的相对平移  t_loop_cur
          double relative_yaw = (*it)->getLoopRelativeYaw();    // 获得关键帧和闭环帧之间的相对偏航角 yaw_loop_cur
          
          // 将 关键帧在闭环时刻和闭环帧的相对位姿（PnP计算的观测值） 和 当前关键帧和闭环帧之间实际的相对位姿（估计值） 构建误差
          ceres::CostFunction* cost_function = FourDOFWeightError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                          relative_yaw, euler_conncected.y(), euler_conncected.z());
          problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index],
                                   t_array[connected_index],
                                   euler_array[i],
                                   t_array[i]);
        }


        if ((*it)->index == cur_index)
          break;
        i++;
      }
      m_keyframelist.unlock();

      ceres::Solve(options, &problem, &summary);    // 求解优化问题

      // 优化完成，使用优化后的位姿来更新关键帧列表中 index 大于等于 first_looped_index 的所有关键帧的位姿
      m_keyframelist.lock();
      i = 0;
      // 用优化结果更新关键帧位姿
      for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
        if ((*it)->index < first_looped_index)
          continue;
        Quaterniond tmp_q;
        tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
        Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
        Matrix3d tmp_r = tmp_q.toRotationMatrix();
        (*it)->updatePose(tmp_t, tmp_r);    // T_w0_b

        if ((*it)->index == cur_index)
          break;
        i++;
      }
      
      // 根据计算出当前关键帧在优化前后的drift，T_w0_w1
      Vector3d cur_t, vio_t;
      Matrix3d cur_r, vio_r;
      cur_kf->getPose(cur_t, cur_r);        // 获取优化后当前关键帧的位姿 T_w0_cur  
      cur_kf->getVioPose(vio_t, vio_r);     // 获取优化前当前关键帧的位姿 T_w1_cur 
      m_drift.lock();
      // 计算 T_w0_w1 只考虑偏航角变化
      yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
      r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
      t_drift = cur_t - r_drift * vio_t;      // t_w0_w1 = t_w0_cur - R_w0_cur * R_w1_cur^T * t_w1_cur = t_w0_cur - r_drift * t_w1_cur
      m_drift.unlock();

      it++;

      // 把当前关键帧it之后的关键帧的位姿通过 T_w0_w1 转换到w0坐标系下
      for (; it != keyframelist.end(); it++) {
        Vector3d P;
        Matrix3d R;
        (*it)->getVioPose(P, R);
        P = r_drift * P + t_drift;
        R = r_drift * R;
        (*it)->updatePose(P, R);
      }
      m_keyframelist.unlock();
      // 优化完后更新path
      updatePath();
    }

    std::chrono::milliseconds dura(2000);
    std::this_thread::sleep_for(dura);
  }
}

void PoseGraph::updatePath() {
  m_keyframelist.lock();
  list<KeyFrame*>::iterator it;
  for (int i = 1; i <= sequence_cnt; i++) {
    path[i].poses.clear();
  }
  base_path.poses.clear();
  posegraph_visualization->reset();

  if (SAVE_LOOP_PATH) {
    ofstream loop_path_file_tmp(VINS_RESULT_PATH, ios::out);
    loop_path_file_tmp.close();
  }

  for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
    Vector3d P;
    Matrix3d R;
    (*it)->getPose(P, R);
    Quaterniond Q;
    Q = R;
    //        printf("path p: %f, %f, %f\n",  P.x(),  P.z(),  P.y() );

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time((*it)->time_stamp);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = P.x() + VISUALIZATION_SHIFT_X;
    pose_stamped.pose.position.y = P.y() + VISUALIZATION_SHIFT_Y;
    pose_stamped.pose.position.z = P.z();
    pose_stamped.pose.orientation.x = Q.x();
    pose_stamped.pose.orientation.y = Q.y();
    pose_stamped.pose.orientation.z = Q.z();
    pose_stamped.pose.orientation.w = Q.w();
    if ((*it)->sequence == 0) {
      base_path.poses.push_back(pose_stamped);
      base_path.header = pose_stamped.header;
    } else {
      path[(*it)->sequence].poses.push_back(pose_stamped);
      path[(*it)->sequence].header = pose_stamped.header;
    }

    if (SAVE_LOOP_PATH) {
      ofstream loop_path_file(VINS_RESULT_PATH, ios::app);
      loop_path_file.setf(ios::fixed, ios::floatfield);
      loop_path_file.precision(10);
      loop_path_file << (*it)->time_stamp << " ";
      loop_path_file.precision(5);
      loop_path_file << P.x() << " "
                     << P.y() << " "
                     << P.z() << " "
                     << Q.x() << " "
                     << Q.y() << " "
                     << Q.z() << " "
                     << Q.w() << endl;
      loop_path_file.close();
    }
    //draw local connection
    if (SHOW_S_EDGE) {
      list<KeyFrame*>::reverse_iterator rit = keyframelist.rbegin();
      list<KeyFrame*>::reverse_iterator lrit;
      for (; rit != keyframelist.rend(); rit++) {
        if ((*rit)->index == (*it)->index) {
          lrit = rit;
          lrit++;
          for (int i = 0; i < 4; i++) {
            if (lrit == keyframelist.rend())
              break;
            if ((*lrit)->sequence == (*it)->sequence) {
              Vector3d conncected_P;
              Matrix3d connected_R;
              (*lrit)->getPose(conncected_P, connected_R);
              posegraph_visualization->add_edge(P, conncected_P);
            }
            lrit++;
          }
          break;
        }
      }
    }
    if (SHOW_L_EDGE) {
      if ((*it)->has_loop && (*it)->sequence == sequence_cnt) {
        KeyFrame* connected_KF = getKeyFrame((*it)->loop_index);
        Vector3d connected_P;
        Matrix3d connected_R;
        connected_KF->getPose(connected_P, connected_R);
        //(*it)->getVioPose(P, R);
        (*it)->getPose(P, R);
        if ((*it)->sequence > 0) {
          posegraph_visualization->add_loopedge(P, connected_P + Vector3d(VISUALIZATION_SHIFT_X, VISUALIZATION_SHIFT_Y, 0));
        }
      }
    }
  }
  publish();
  m_keyframelist.unlock();
}

void PoseGraph::savePoseGraph() {
  m_keyframelist.lock();
  TicToc tmp_t;
  FILE* pFile;
  printf("pose graph path: %s\n", POSE_GRAPH_SAVE_PATH.c_str());
  printf("pose graph saving... \n");
  string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
  pFile = fopen(file_path.c_str(), "w");
  //fprintf(pFile, "index time_stamp Tx Ty Tz Qw Qx Qy Qz loop_index loop_info\n");
  list<KeyFrame*>::iterator it;
  for (it = keyframelist.begin(); it != keyframelist.end(); it++) {
    std::string image_path, descriptor_path, brief_path, keypoints_path;
    if (DEBUG_IMAGE) {
      image_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_image.png";
      imwrite(image_path.c_str(), (*it)->image);
    }
    Quaterniond VIO_tmp_Q{(*it)->vio_R_w_i};
    Quaterniond PG_tmp_Q{(*it)->R_w_i};
    Vector3d VIO_tmp_T = (*it)->vio_T_w_i;
    Vector3d PG_tmp_T = (*it)->T_w_i;

    fprintf(pFile, " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d\n", (*it)->index, (*it)->time_stamp,
            VIO_tmp_T.x(), VIO_tmp_T.y(), VIO_tmp_T.z(),
            PG_tmp_T.x(), PG_tmp_T.y(), PG_tmp_T.z(),
            VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(),
            PG_tmp_Q.w(), PG_tmp_Q.x(), PG_tmp_Q.y(), PG_tmp_Q.z(),
            (*it)->loop_index,
            (*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2), (*it)->loop_info(3),
            (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6), (*it)->loop_info(7),
            (int)(*it)->keypoints.size());

    // write keypoints, brief_descriptors   vector<cv::KeyPoint> keypoints vector<BRIEF::bitset> brief_descriptors;
    assert((*it)->keypoints.size() == (*it)->brief_descriptors.size());
    brief_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_briefdes.dat";
    std::ofstream brief_file(brief_path, std::ios::binary);
    keypoints_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_keypoints.txt";
    FILE* keypoints_file;
    keypoints_file = fopen(keypoints_path.c_str(), "w");
    for (int i = 0; i < (int)(*it)->keypoints.size(); i++) {
      brief_file << (*it)->brief_descriptors[i] << endl;
      fprintf(keypoints_file, "%f %f %f %f\n", (*it)->keypoints[i].pt.x, (*it)->keypoints[i].pt.y,
              (*it)->keypoints_norm[i].pt.x, (*it)->keypoints_norm[i].pt.y);
    }
    brief_file.close();
    fclose(keypoints_file);
  }
  fclose(pFile);

  printf("save pose graph time: %f s\n", tmp_t.toc() / 1000);
  m_keyframelist.unlock();
}


/**
 * @brief 读取PoseGraph
 */
void PoseGraph::loadPoseGraph() {
  TicToc tmp_t;
  FILE* pFile;
  string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
  printf("lode pose graph from: %s \n", file_path.c_str());
  printf("pose graph loading...\n");
  pFile = fopen(file_path.c_str(), "r");
  if (pFile == NULL) {
    printf("lode previous pose graph error: wrong previous pose graph path or no previous pose graph \n the system will start with new pose graph \n");
    return;
  }
  int index;
  double time_stamp;
  double VIO_Tx, VIO_Ty, VIO_Tz;
  double PG_Tx, PG_Ty, PG_Tz;
  double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
  double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
  double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
  double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
  int loop_index;
  int keypoints_num;
  Eigen::Matrix<double, 8, 1> loop_info;
  int cnt = 0;
  while (fscanf(pFile, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d", &index, &time_stamp,
                &VIO_Tx, &VIO_Ty, &VIO_Tz,
                &PG_Tx, &PG_Ty, &PG_Tz,
                &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz,
                &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz,
                &loop_index,
                &loop_info_0, &loop_info_1, &loop_info_2, &loop_info_3,
                &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7,
                &keypoints_num) != EOF) {
    /*
        printf("I read: %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d\n", index, time_stamp, 
                                    VIO_Tx, VIO_Ty, VIO_Tz, 
                                    PG_Tx, PG_Ty, PG_Tz, 
                                    VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz, 
                                    PG_Qw, PG_Qx, PG_Qy, PG_Qz, 
                                    loop_index,
                                    loop_info_0, loop_info_1, loop_info_2, loop_info_3, 
                                    loop_info_4, loop_info_5, loop_info_6, loop_info_7,
                                    keypoints_num);
        */
    cv::Mat image;
    std::string image_path, descriptor_path;
    if (DEBUG_IMAGE) {
      image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
      image = cv::imread(image_path.c_str(), 0);
    }

    Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
    Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
    Quaterniond VIO_Q;
    VIO_Q.w() = VIO_Qw;
    VIO_Q.x() = VIO_Qx;
    VIO_Q.y() = VIO_Qy;
    VIO_Q.z() = VIO_Qz;
    Quaterniond PG_Q;
    PG_Q.w() = PG_Qw;
    PG_Q.x() = PG_Qx;
    PG_Q.y() = PG_Qy;
    PG_Q.z() = PG_Qz;
    Matrix3d VIO_R, PG_R;
    VIO_R = VIO_Q.toRotationMatrix();
    PG_R = PG_Q.toRotationMatrix();
    Eigen::Matrix<double, 8, 1> loop_info;
    loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6, loop_info_7;

    if (loop_index != -1)
      if (earliest_loop_index > loop_index || earliest_loop_index == -1) {
        earliest_loop_index = loop_index;
      }

    // load keypoints, brief_descriptors
    string brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_briefdes.dat";
    std::ifstream brief_file(brief_path, std::ios::binary);
    string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_keypoints.txt";
    FILE* keypoints_file;
    keypoints_file = fopen(keypoints_path.c_str(), "r");
    vector<cv::KeyPoint> keypoints;
    vector<cv::KeyPoint> keypoints_norm;
    vector<BRIEF::bitset> brief_descriptors;
    for (int i = 0; i < keypoints_num; i++) {
      BRIEF::bitset tmp_des;
      brief_file >> tmp_des;
      brief_descriptors.push_back(tmp_des);
      cv::KeyPoint tmp_keypoint;
      cv::KeyPoint tmp_keypoint_norm;
      double p_x, p_y, p_x_norm, p_y_norm;
      if (!fscanf(keypoints_file, "%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm))
        printf(" fail to load pose graph \n");
      tmp_keypoint.pt.x = p_x;
      tmp_keypoint.pt.y = p_y;
      tmp_keypoint_norm.pt.x = p_x_norm;
      tmp_keypoint_norm.pt.y = p_y_norm;
      keypoints.push_back(tmp_keypoint);
      keypoints_norm.push_back(tmp_keypoint_norm);
    }
    brief_file.close();
    fclose(keypoints_file);
    // 构造地图关键帧
    KeyFrame* keyframe = new KeyFrame(time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, loop_index, loop_info, keypoints, keypoints_norm, brief_descriptors);
    loadKeyFrame(keyframe, 0);
    if (cnt % 20 == 0) {
      publish();
    }
    cnt++;
  }
  fclose(pFile);
  printf("load pose graph time: %f s\n", tmp_t.toc() / 1000);
  base_sequence = 0;
}

void PoseGraph::publish() {
  for (int i = 1; i <= sequence_cnt; i++) {
    //if (sequence_loop[i] == true || i == base_sequence)
    if (1 || i == base_sequence) {
      pub_pg_path.publish(path[i]);
      pub_path[i].publish(path[i]);
      posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
    }
  }
  base_path.header.frame_id = "world";
  pub_base_path.publish(base_path);
  //posegraph_visualization->publish_by(pub_pose_graph, path[sequence_cnt].header);
}

/**
 * @brief 更新 index闭环帧和对应关键帧的信息
 * 
 * @param[in] index         闭环帧在关键帧列表中的索引
 * @param[in] _loop_info    闭环帧和对应关键帧的 T_loop_cur
 */
void PoseGraph::updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1>& _loop_info) {
  KeyFrame* kf = getKeyFrame(index);
  kf->updateLoop(_loop_info);
  if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
    if (FAST_RELOCALIZATION) {
      KeyFrame* old_kf = getKeyFrame(kf->loop_index);   // 获得闭环帧
      Vector3d w_P_old, w_P_cur, vio_P_cur;
      Matrix3d w_R_old, w_R_cur, vio_R_cur;
      old_kf->getPose(w_P_old, w_R_old);      // 获得闭环帧在PoseGraph中的位姿 T_w0_loop
      kf->getVioPose(vio_P_cur, vio_R_cur);   // 获得当前关键帧在轨迹中的位姿 T_w1_cur

      Vector3d relative_t;
      Quaterniond relative_q;
      relative_t = kf->getLoopRelativeT();                        // t_loop_cur
      relative_q = (kf->getLoopRelativeQ()).toRotationMatrix();   // R_loop_cur

      // 计算 T_w0_cur = T_w0_loop * T_loop_cur
      w_P_cur = w_R_old * relative_t + w_P_old;
      w_R_cur = w_R_old * relative_q;

      double shift_yaw;
      Matrix3d shift_r;
      Vector3d shift_t;
      // 计算 T_w0_w1 = T_w0_cur * T_w1_cur^{-1}
      shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
      shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
      shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;
      
      // 更新 T_w0_w1
      m_drift.lock();
      yaw_drift = shift_yaw;
      r_drift = shift_r;
      t_drift = shift_t;
      m_drift.unlock();
    }
  }
}