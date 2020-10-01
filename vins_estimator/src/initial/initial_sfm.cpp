#include "initial_sfm.h"

GlobalSFM::GlobalSFM() {}

/**
 * @brief 利用两帧相机位姿，进行线性三角化，求解特征点的世界坐标
 * 
 * @param[in] Pose0       相机0位姿 T_0w
 * @param[in] Pose1       相机1位姿 T_1w
 * @param[in] point0      特征点在相机0的归一化坐标
 * @param[in] point1      特征点在相机1的归一化坐标
 * @param[out] point_3d   特征点的世界坐标
 */
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                                 Vector2d &point0, Vector2d &point1, Vector3d &point_3d) {
  Matrix4d design_matrix = Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Vector4d triangulated_point;
  triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();   // SVD求解超定方程组
  // 齐次坐标 转为 非齐次坐标
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


/**
 * @brief 
 * 
 * @param[in] R_initial
 * @param[in] P_initial
 * @param[in] i
 * @param[in] sfm_f
 * @return true 
 * @return false 
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f) {
  vector<cv::Point2f> pts_2_vector;
  vector<cv::Point3f> pts_3_vector;
  for (int j = 0; j < feature_num; j++) {
    if (sfm_f[j].state != true)
      continue;
    Vector2d point2d;
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
      if (sfm_f[j].observation[k].first == i) {
        Vector2d img_pts = sfm_f[j].observation[k].second;
        cv::Point2f pts_2(img_pts(0), img_pts(1));
        pts_2_vector.push_back(pts_2);
        cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
        pts_3_vector.push_back(pts_3);
        break;
      }
    }
  }
  if (int(pts_2_vector.size()) < 15) {
    printf("unstable features tracking, please slowly move you device!\n");
    if (int(pts_2_vector.size()) < 10)
      return false;
  }
  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
  if (!pnp_succ) {
    return false;
  }
  cv::Rodrigues(rvec, r);
  //cout << "r " << endl << r << endl;
  MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  R_initial = R_pnp;
  P_initial = T_pnp;
  return true;
}


/**
 * @brief 对两帧图像的共视点进行三角化
 * 
 * @param[in] frame0    图像0
 * @param[in] Pose0     图像0的位姿T_l0
 * @param[in] frame1    图像1
 * @param[in] Pose1     图像1的位姿T_l1
 * @param[in] sfm_f     滑窗中的所有特征点
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                                     int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                                     vector<SFMFeature> &sfm_f) {
  assert(frame0 != frame1);
  // 遍历 所有特征
  for (int j = 0; j < feature_num; j++) {
    if (sfm_f[j].state == true)   // 如果当前特征已经三角化过了，则跳过
      continue;
    bool has_0 = false, has_1 = false;  // 
    Vector2d point0;
    Vector2d point1;
    // 遍历当前特征的所有观测
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {  
      if (sfm_f[j].observation[k].first == frame0) {
        point0 = sfm_f[j].observation[k].second;    // 获得 在图像0的归一化坐标前两维
        has_0 = true;
      }
      if (sfm_f[j].observation[k].first == frame1) {
        point1 = sfm_f[j].observation[k].second;    // 获得 在图像1的归一化坐标前两维
        has_1 = true;                                
      }
    }
    if (has_0 && has_1) {   // 如果当前特征被
      Vector3d point_3d;
      triangulatePoint(Pose0, Pose1, point0, point1, point_3d);   // 三角化当前特征，获得 Pw
      sfm_f[j].state = true;
      sfm_f[j].position[0] = point_3d(0);
      sfm_f[j].position[1] = point_3d(1);
      sfm_f[j].position[2] = point_3d(2);
      //cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
    }
  }
}


/**
 * @brief 纯视觉BA，求解窗口中的所有图像帧的位姿和特征点坐标，假设情况如下:
 * 
 * 					|      |   |   |   |    |     |   |   |   |     |  
 *          0      1   2   3   4    5     6   7   8   9     10
 *         第一帧                  参考帧l                  当前帧c 
 * 				
 * Step 1: 三角化 参考帧和当前帧的共视点
 * Step 2: PnP 计算[6-9]的位姿
 * Step 3: 三角化 [6-9] 和 当前帧 的共视点
 * Step 4: 三角化 参考帧 和 [6-9] 的共视点
 * Step 5: PnP 计算[0-4]的位姿
 * Step 6: 三角化 [0-4] 和 参考帧 的共视点
 * Step 7: 三角化 其他未恢复的点
 * Step 8: 全局BA，优化变量为滑窗内所有相机位姿和路标点
 * 
 * @param[in]   frame_num		滑窗内图像帧数目
 * @param[out]  q 					滑窗内图像帧的旋转四元数Q（相对于第l帧）
 * @param[out]	T 					滑窗内图像帧的平移向量t（相对于第l帧）
 * @param[in]  	l 					第l帧，作为参考帧，设为世界坐标系
 * @param[in]  	relative_R	当前帧到第l帧的旋转矩阵  R_cl
 * @param[in]  	relative_T 	当前帧到第l帧的平移向量  t_cl
 * @param[in]  	sfm_f				所有特征点
 * @param[out]  sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * 
 * @return  bool true				sfm求解成功
 * 					bool false			sfm求解失败
 */
bool GlobalSFM::construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                          const Matrix3d relative_R, const Vector3d relative_T,
                          vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points) {
  feature_num = sfm_f.size();		// 获得特征数目
  
	// 假设第l帧为原点，根据当前帧到第l帧的relative_R，relative_T，得到当前帧位姿
	q[l].w() = 1;
  q[l].x() = 0;
  q[l].y() = 0;
  q[l].z() = 0;
  T[l].setZero();
  q[frame_num - 1] = q[l] * Quaterniond(relative_R);	// R_cl
  T[frame_num - 1] = relative_T;                      // t_cl  
  //cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
  //cout << "init t_l " << T[l].transpose() << endl;

  // 滑窗内每一帧在参考帧坐标系下的位姿
  Matrix3d c_Rotation[frame_num];     // R_li 旋转矩阵形式
  Vector3d c_Translation[frame_num];  // t_li
  Quaterniond c_Quat[frame_num];      // R_li 四元数形式
  double c_rotation[frame_num][4];    // R_li 数组形式，用于ceres
  double c_translation[frame_num][3]; // t_li 数组形式，用于ceres
  Eigen::Matrix<double, 3, 4> Pose[frame_num];		// 这里的pose表示的是 第l帧到 每一帧 的变换矩阵

  c_Quat[l] = q[l].inverse();		// R_ll
  c_Rotation[l] = c_Quat[l].toRotationMatrix();
  c_Translation[l] = -1 * (c_Rotation[l] * T[l]);		// t_ll
  Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
  Pose[l].block<3, 1>(0, 3) = c_Translation[l];

  c_Quat[frame_num - 1] = q[frame_num - 1].inverse();		// R_lc
  c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();	
  c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);	// t_lc
  Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
  Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

  // 1、先三角化 参考帧l 与 第当前帧c 的共视路标点
  // 2、pnp 求解 [l+1, c-1] 与 l 的变换矩阵 R_initial、P_initial，保存在Pose中
	// 3、三角化 [l+1, c-1] 与 c 之间的共视路标点
  for (int i = l; i < frame_num - 1; i++) {
    // solve pnp
    if (i > l) {
      Matrix3d R_initial = c_Rotation[i - 1];
      Vector3d P_initial = c_Translation[i - 1];
      if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
        return false;
      c_Rotation[i] = R_initial;
      c_Translation[i] = P_initial;
      c_Quat[i] = c_Rotation[i];
      Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
      Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    // triangulate point based on the solve pnp result
    triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
  }

  // 4、对 l 与 [l+1, c-1] 的每一帧再进行三角化
  for (int i = l + 1; i < frame_num - 1; i++)
    triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	

  // 5、PNP求解 [0, l-1] 与 l 之间的变换矩阵
	// 6、三角化 [0, l-1] 与 l 之间共视路标点
  for (int i = l - 1; i >= 0; i--) {
    //solve pnp
    Matrix3d R_initial = c_Rotation[i + 1];
    Vector3d P_initial = c_Translation[i + 1];
    if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
      return false;
    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    c_Quat[i] = c_Rotation[i];
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    //triangulate
    triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
  }

  // 7、三角化其他未恢复的特征点
  for (int j = 0; j < feature_num; j++) {
    if (sfm_f[j].state == true) {		// 如果已经三角化，则跳过
			continue;
		}

    if ((int)sfm_f[j].observation.size() >= 2) {	// 三角化要求空间点至少被两帧相机观测到
      Vector2d point0, point1;
      int frame_0 = sfm_f[j].observation[0].first;			// 特征在滑窗中的第一个观测
      point0 = sfm_f[j].observation[0].second;
      int frame_1 = sfm_f[j].observation.back().first;	// 特征在滑窗中的最后一个观测
      point1 = sfm_f[j].observation.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);		// 三角化
      sfm_f[j].state = true;
      sfm_f[j].position[0] = point_3d(0);
      sfm_f[j].position[1] = point_3d(1);
      sfm_f[j].position[2] = point_3d(2);
      //cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
    }
  }
	/** 至此得到了滑动窗口中所有图像帧的位姿以及特征点的3d坐标 **/


  /*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/

  // 全局BA，优化变量为滑窗内所有相机位姿
  ceres::Problem problem;
  ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
  
  // 将每一帧位姿 T_li 转换为 数组形式，用于ceres
  for (int i = 0; i < frame_num; i++) {
    //double array for ceres
    c_translation[i][0] = c_Translation[i].x();
    c_translation[i][1] = c_Translation[i].y();
    c_translation[i][2] = c_Translation[i].z();
    c_rotation[i][0] = c_Quat[i].w();
    c_rotation[i][1] = c_Quat[i].x();
    c_rotation[i][2] = c_Quat[i].y();
    c_rotation[i][3] = c_Quat[i].z();
    problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    problem.AddParameterBlock(c_translation[i], 3);

		// 固定 参考帧c的位姿 和 当前帧c的平移
		// 把 参考帧和当前帧之间的平移量作为单位平移量
    if (i == l) {
      problem.SetParameterBlockConstant(c_rotation[i]);
    }
    if (i == l || i == frame_num - 1) {
      problem.SetParameterBlockConstant(c_translation[i]);
    }
  }

  for (int i = 0; i < feature_num; i++) {
    if (sfm_f[i].state != true)
      continue;
    for (int j = 0; j < int(sfm_f[i].observation.size()); j++) {
      int l = sfm_f[i].observation[j].first;
      ceres::CostFunction *cost_function = ReprojectionError3D::Create(
          sfm_f[i].observation[j].second.x(),
          sfm_f[i].observation[j].second.y());

      problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],
                               sfm_f[i].position);
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;		// 采用稠密矩阵求解器
  //options.minimizer_progress_to_stdout = true;	
  options.max_solver_time_in_seconds = 0.2;						// 最大优化时间为 0.2s
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //std::cout << summary.BriefReport() << "\n";
  if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03) {
    //cout << "vision only BA converge" << endl;
  } else {
    //cout << "vision only BA not converge " << endl;
    return false;
  }

	// 还原出 优化变量R_li
  for (int i = 0; i < frame_num; i++) {
    q[i].w() = c_rotation[i][0];
    q[i].x() = c_rotation[i][1];
    q[i].y() = c_rotation[i][2];
    q[i].z() = c_rotation[i][3];
    q[i] = q[i].inverse();
    //cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
  }

	// 还原出 优化变量t_li
  for (int i = 0; i < frame_num; i++) {
    T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    //cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
  }

	// 还原出 优化变量特征3d点
  for (int i = 0; i < (int)sfm_f.size(); i++) {
    if (sfm_f[i].state)
      sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
  }

  return true;
}
