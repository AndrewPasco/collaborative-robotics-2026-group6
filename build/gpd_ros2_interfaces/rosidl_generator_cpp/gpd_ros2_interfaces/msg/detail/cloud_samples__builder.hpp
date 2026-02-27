// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gpd_ros2_interfaces:msg/CloudSamples.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SAMPLES__BUILDER_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SAMPLES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gpd_ros2_interfaces/msg/detail/cloud_samples__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gpd_ros2_interfaces
{

namespace msg
{

namespace builder
{

class Init_CloudSamples_samples
{
public:
  explicit Init_CloudSamples_samples(::gpd_ros2_interfaces::msg::CloudSamples & msg)
  : msg_(msg)
  {}
  ::gpd_ros2_interfaces::msg::CloudSamples samples(::gpd_ros2_interfaces::msg::CloudSamples::_samples_type arg)
  {
    msg_.samples = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::CloudSamples msg_;
};

class Init_CloudSamples_cloud_sources
{
public:
  Init_CloudSamples_cloud_sources()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CloudSamples_samples cloud_sources(::gpd_ros2_interfaces::msg::CloudSamples::_cloud_sources_type arg)
  {
    msg_.cloud_sources = std::move(arg);
    return Init_CloudSamples_samples(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::CloudSamples msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::gpd_ros2_interfaces::msg::CloudSamples>()
{
  return gpd_ros2_interfaces::msg::builder::Init_CloudSamples_cloud_sources();
}

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SAMPLES__BUILDER_HPP_
