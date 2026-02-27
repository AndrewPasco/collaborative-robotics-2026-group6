// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gpd_ros2_interfaces:msg/CloudSources.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SOURCES__BUILDER_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SOURCES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gpd_ros2_interfaces/msg/detail/cloud_sources__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gpd_ros2_interfaces
{

namespace msg
{

namespace builder
{

class Init_CloudSources_view_points
{
public:
  explicit Init_CloudSources_view_points(::gpd_ros2_interfaces::msg::CloudSources & msg)
  : msg_(msg)
  {}
  ::gpd_ros2_interfaces::msg::CloudSources view_points(::gpd_ros2_interfaces::msg::CloudSources::_view_points_type arg)
  {
    msg_.view_points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::CloudSources msg_;
};

class Init_CloudSources_camera_source
{
public:
  explicit Init_CloudSources_camera_source(::gpd_ros2_interfaces::msg::CloudSources & msg)
  : msg_(msg)
  {}
  Init_CloudSources_view_points camera_source(::gpd_ros2_interfaces::msg::CloudSources::_camera_source_type arg)
  {
    msg_.camera_source = std::move(arg);
    return Init_CloudSources_view_points(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::CloudSources msg_;
};

class Init_CloudSources_cloud
{
public:
  Init_CloudSources_cloud()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CloudSources_camera_source cloud(::gpd_ros2_interfaces::msg::CloudSources::_cloud_type arg)
  {
    msg_.cloud = std::move(arg);
    return Init_CloudSources_camera_source(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::CloudSources msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::gpd_ros2_interfaces::msg::CloudSources>()
{
  return gpd_ros2_interfaces::msg::builder::Init_CloudSources_cloud();
}

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SOURCES__BUILDER_HPP_
