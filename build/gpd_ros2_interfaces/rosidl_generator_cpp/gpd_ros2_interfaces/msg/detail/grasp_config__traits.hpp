// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from gpd_ros2_interfaces:msg/GraspConfig.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__TRAITS_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "gpd_ros2_interfaces/msg/detail/grasp_config__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'position'
// Member 'sample'
#include "geometry_msgs/msg/detail/point__traits.hpp"
// Member 'approach'
// Member 'binormal'
// Member 'axis'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace gpd_ros2_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const GraspConfig & msg,
  std::ostream & out)
{
  out << "{";
  // member: position
  {
    out << "position: ";
    to_flow_style_yaml(msg.position, out);
    out << ", ";
  }

  // member: approach
  {
    out << "approach: ";
    to_flow_style_yaml(msg.approach, out);
    out << ", ";
  }

  // member: binormal
  {
    out << "binormal: ";
    to_flow_style_yaml(msg.binormal, out);
    out << ", ";
  }

  // member: axis
  {
    out << "axis: ";
    to_flow_style_yaml(msg.axis, out);
    out << ", ";
  }

  // member: width
  {
    out << "width: ";
    rosidl_generator_traits::value_to_yaml(msg.width, out);
    out << ", ";
  }

  // member: score
  {
    out << "score: ";
    rosidl_generator_traits::value_to_yaml(msg.score, out);
    out << ", ";
  }

  // member: sample
  {
    out << "sample: ";
    to_flow_style_yaml(msg.sample, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GraspConfig & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position:\n";
    to_block_style_yaml(msg.position, out, indentation + 2);
  }

  // member: approach
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "approach:\n";
    to_block_style_yaml(msg.approach, out, indentation + 2);
  }

  // member: binormal
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "binormal:\n";
    to_block_style_yaml(msg.binormal, out, indentation + 2);
  }

  // member: axis
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "axis:\n";
    to_block_style_yaml(msg.axis, out, indentation + 2);
  }

  // member: width
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "width: ";
    rosidl_generator_traits::value_to_yaml(msg.width, out);
    out << "\n";
  }

  // member: score
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "score: ";
    rosidl_generator_traits::value_to_yaml(msg.score, out);
    out << "\n";
  }

  // member: sample
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sample:\n";
    to_block_style_yaml(msg.sample, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GraspConfig & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace gpd_ros2_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use gpd_ros2_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const gpd_ros2_interfaces::msg::GraspConfig & msg,
  std::ostream & out, size_t indentation = 0)
{
  gpd_ros2_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gpd_ros2_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const gpd_ros2_interfaces::msg::GraspConfig & msg)
{
  return gpd_ros2_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<gpd_ros2_interfaces::msg::GraspConfig>()
{
  return "gpd_ros2_interfaces::msg::GraspConfig";
}

template<>
inline const char * name<gpd_ros2_interfaces::msg::GraspConfig>()
{
  return "gpd_ros2_interfaces/msg/GraspConfig";
}

template<>
struct has_fixed_size<gpd_ros2_interfaces::msg::GraspConfig>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Point>::value && has_fixed_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct has_bounded_size<gpd_ros2_interfaces::msg::GraspConfig>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Point>::value && has_bounded_size<geometry_msgs::msg::Vector3>::value> {};

template<>
struct is_message<gpd_ros2_interfaces::msg::GraspConfig>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG__TRAITS_HPP_
