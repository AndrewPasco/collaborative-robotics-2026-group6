// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gpd_ros2_interfaces:msg/GraspConfigList.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__BUILDER_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gpd_ros2_interfaces
{

namespace msg
{

namespace builder
{

class Init_GraspConfigList_grasps
{
public:
  explicit Init_GraspConfigList_grasps(::gpd_ros2_interfaces::msg::GraspConfigList & msg)
  : msg_(msg)
  {}
  ::gpd_ros2_interfaces::msg::GraspConfigList grasps(::gpd_ros2_interfaces::msg::GraspConfigList::_grasps_type arg)
  {
    msg_.grasps = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfigList msg_;
};

class Init_GraspConfigList_header
{
public:
  Init_GraspConfigList_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GraspConfigList_grasps header(::gpd_ros2_interfaces::msg::GraspConfigList::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_GraspConfigList_grasps(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfigList msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::gpd_ros2_interfaces::msg::GraspConfigList>()
{
  return gpd_ros2_interfaces::msg::builder::Init_GraspConfigList_header();
}

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__BUILDER_HPP_
