#include "utility.h"

/**
 * @brief 计算 g和重力向量之间的旋转量
 * 
 * @param[in] g     向量
 * @return Eigen::Matrix3d  g 和 重力向量之间的旋转矩阵 R_Gg  在这里可以认为是 R_wb
 */
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();       // 单位化
    Eigen::Vector3d ng2{0, 0, 1.0};             // 重力方向
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();        // 计算偏航角
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;      // 设置偏航角为0
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
