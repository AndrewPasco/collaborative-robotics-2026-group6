// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gpd_ros2_interfaces:msg/GraspConfig.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__BUILDER_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gpd_ros2_interfaces/msg/detail/grasp_config__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gpd_ros2_interfaces
{

namespace msg
{

namespace builder
{

class Init_GraspConfig_sample
{
public:
  explicit Init_GraspConfig_sample(::gpd_ros2_interfaces::msg::GraspConfig & msg)
  : msg_(msg)
  {}
  ::gpd_ros2_interfaces::msg::GraspConfig sample(::gpd_ros2_interfaces::msg::GraspConfig::_sample_type arg)
  {
    msg_.sample = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfig msg_;
};

class Init_GraspConfig_score
{
public:
  explicit Init_GraspConfig_score(::gpd_ros2_interfaces::msg::GraspConfig & msg)
  : msg_(msg)
  {}
  Init_GraspConfig_sample score(::gpd_ros2_interfaces::msg::GraspConfig::_score_type arg)
  {
    msg_.score = std::move(arg);
    return Init_GraspConfig_sample(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfig msg_;
};

class Init_GraspConfig_width
{
public:
  explicit Init_GraspConfig_width(::gpd_ros2_interfaces::msg::GraspConfig & msg)
  : msg_(msg)
  {}
  Init_GraspConfig_score width(::gpd_ros2_interfaces::msg::GraspConfig::_width_type arg)
  {
    msg_.width = std::move(arg);
    return Init_GraspConfig_score(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfig msg_;
};

class Init_GraspConfig_axis
{
public:
  explicit Init_GraspConfig_axis(::gpd_ros2_interfaces::msg::GraspConfig & msg)
  : msg_(msg)
  {}
  Init_GraspConfig_width axis(::gpd_ros2_interfaces::msg::GraspConfig::_axis_type arg)
  {
    msg_.axis = std::move(arg);
    return Init_GraspConfig_width(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfig msg_;
};

class Init_GraspConfig_binormal
{
public:
  explicit Init_GraspConfig_binormal(::gpd_ros2_interfaces::msg::GraspConfig & msg)
  : msg_(msg)
  {}
  Init_GraspConfig_axis binormal(::gpd_ros2_interfaces::msg::GraspConfig::_binormal_type arg)
  {
    msg_.binormal = std::move(arg);
    return Init_GraspConfig_axis(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfig msg_;
};

class Init_GraspConfig_approach
{
public:
  explicit Init_GraspConfig_approach(::gpd_ros2_interfaces::msg::GraspConfig & msg)
  : msg_(msg)
  {}
  Init_GraspConfig_binormal approach(::gpd_ros2_interfaces::msg::GraspConfig::_approach_type arg)
  {
    msg_.approach = std::move(arg);
    return Init_GraspConfig_binormal(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfig msg_;
};

class Init_GraspConfig_position
{
public:
  Init_GraspConfig_position()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GraspConfig_approach position(::gpd_ros2_interfaces::msg::GraspConfig::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_GraspConfig_approach(msg_);
  }

private:
  ::gpd_ros2_interfaces::msg::GraspConfig msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::gpd_ros2_interfaces::msg::GraspConfig>()
{
  return gpd_ros2_interfaces::msg::builder::Init_GraspConfig_position();
}

}  // namespace gpd_ros2_interfaces

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__BUILDER_HPP_
