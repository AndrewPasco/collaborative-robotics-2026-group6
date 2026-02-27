// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from gpd_ros2_interfaces:msg/GraspConfigList.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__TRAITS_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'grasps'
#include "gpd_ros2_interfaces/msg/detail/grasp_config__traits.hpp"

namespace gpd_ros2_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const GraspConfigList & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: grasps
  {
    if (msg.grasps.size() == 0) {
      out << "grasps: []";
    } else {
      out << "grasps: [";
      size_t pending_items = msg.grasps.size();
      for (auto item : msg.grasps) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GraspConfigList & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: grasps
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.grasps.size() == 0) {
      out << "grasps: []\n";
    } else {
      out << "grasps:\n";
      for (auto item : msg.grasps) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GraspConfigList & msg, bool use_flow_style = false)
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
  const gpd_ros2_interfaces::msg::GraspConfigList & msg,
  std::ostream & out, size_t indentation = 0)
{
  gpd_ros2_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gpd_ros2_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const gpd_ros2_interfaces::msg::GraspConfigList & msg)
{
  return gpd_ros2_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<gpd_ros2_interfaces::msg::GraspConfigList>()
{
  return "gpd_ros2_interfaces::msg::GraspConfigList";
}

template<>
inline const char * name<gpd_ros2_interfaces::msg::GraspConfigList>()
{
  return "gpd_ros2_interfaces/msg/GraspConfigList";
}

template<>
struct has_fixed_size<gpd_ros2_interfaces::msg::GraspConfigList>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<gpd_ros2_interfaces::msg::GraspConfigList>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<gpd_ros2_interfaces::msg::GraspConfigList>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__TRAITS_HPP_
