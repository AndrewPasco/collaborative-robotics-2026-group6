// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gpd_ros2_interfaces:msg/CloudIndexed.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__BUILDER_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gpd_ros2_interfaces
{

namespace msg
{

namespace builder
{

class Init_CloudIndexed_indices
{
public:
  explicit Init_CloudIndexed_indices(::gpd_ros2_interfaces::msg::CloudIndexed & msg)
  : msg_(msg)
  {}
  ::gpd_ros2_interfaces::msg::CloudIndexed indices(::gpd_ros2_interfaces::msg::CloudIndexed::_indices_type arg)
  {
    msg_.indices = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::CloudIndexed msg_;
};

class Init_CloudIndexed_cloud_sources
{
public:
  Init_CloudIndexed_cloud_sources()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CloudIndexed_indices cloud_sources(::gpd_ros2_interfaces::msg::CloudIndexed::_cloud_sources_type arg)
  {
    msg_.cloud_sources = std::move(arg);
    return Init_CloudIndexed_indices(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::CloudIndexed msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::gpd_ros2_interfaces::msg::CloudIndexed>()
{
  return gpd_ros2_interfaces::msg::builder::Init_CloudIndexed_cloud_sources();
}

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__BUILDER_HPP_
