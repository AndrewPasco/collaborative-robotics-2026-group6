// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gpd_ros2_interfaces:msg/SamplesMsg.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__SAMPLES_MSG__BUILDER_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__SAMPLES_MSG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gpd_ros2_interfaces/msg/detail/samples_msg__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gpd_ros2_interfaces
{

namespace msg
{

namespace builder
{

class Init_SamplesMsg_samples
{
public:
  explicit Init_SamplesMsg_samples(::gpd_ros2_interfaces::msg::SamplesMsg & msg)
  : msg_(msg)
  {}
  ::gpd_ros2_interfaces::msg::SamplesMsg samples(::gpd_ros2_interfaces::msg::SamplesMsg::_samples_type arg)
  {
    msg_.samples = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::SamplesMsg msg_;
};

class Init_SamplesMsg_header
{
public:
  Init_SamplesMsg_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SamplesMsg_samples header(::gpd_ros2_interfaces::msg::SamplesMsg::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_SamplesMsg_samples(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::SamplesMsg msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::gpd_ros2_interfaces::msg::SamplesMsg>()
{
  return gpd_ros2_interfaces::msg::builder::Init_SamplesMsg_header();
}

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__SAMPLES_MSG__BUILDER_HPP_
