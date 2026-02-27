// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__BUILDER_HPP_
#define GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gpd_ros2_interfaces/srv/detail/detect_grasps__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gpd_ros2_interfaces
{

namespace srv
{

namespace builder
{

class Init_DetectGrasps_Request_cloud_indexed
{
public:
  Init_DetectGrasps_Request_cloud_indexed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::gpd_ros2_interfaces::srv::DetectGrasps_Request cloud_indexed(::gpd_ros2_interfaces::srv::DetectGrasps_Request::_cloud_indexed_type arg)
  {
    msg_.cloud_indexed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gpd_ros2_interfaces::srv::DetectGrasps_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gpd_ros2_interfaces::srv::DetectGrasps_Request>()
{
  return gpd_ros2_interfaces::srv::builder::Init_DetectGrasps_Request_cloud_indexed();
}

}  // namespace gpd_ros2_interfaces


namespace gpd_ros2_interfaces
{

namespace srv
{

namespace builder
{

class Init_DetectGrasps_Response_grasp_configs
{
public:
  Init_DetectGrasps_Response_grasp_configs()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::gpd_ros2_interfaces::srv::DetectGrasps_Response grasp_configs(::gpd_ros2_interfaces::srv::DetectGrasps_Response::_grasp_configs_type arg)
  {
    msg_.grasp_configs = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gpd_ros2_interfaces::srv::DetectGrasps_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gpd_ros2_interfaces::srv::DetectGrasps_Response>()
{
  return gpd_ros2_interfaces::srv::builder::Init_DetectGrasps_Response_grasp_configs();
}

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__BUILDER_HPP_
