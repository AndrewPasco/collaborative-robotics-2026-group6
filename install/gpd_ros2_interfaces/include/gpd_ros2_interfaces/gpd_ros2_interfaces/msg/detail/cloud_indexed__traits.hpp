// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from gpd_ros2_interfaces:msg/CloudIndexed.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__TRAITS_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'cloud_sources'
#include "gpd_ros2_interfaces/msg/detail/cloud_sources__traits.hpp"

namespace gpd_ros2_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const CloudIndexed & msg,
  std::ostream & out)
{
  out << "{";
  // member: cloud_sources
  {
    out << "cloud_sources: ";
    to_flow_style_yaml(msg.cloud_sources, out);
    out << ", ";
  }

  // member: indices
  {
    if (msg.indices.size() == 0) {
      out << "indices: []";
    } else {
      out << "indices: [";
      size_t pending_items = msg.indices.size();
      for (auto item : msg.indices) {
        rosidl_generator_traits::value_to_yaml(item, out);
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
  const CloudIndexed & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: cloud_sources
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cloud_sources:\n";
    to_block_style_yaml(msg.cloud_sources, out, indentation + 2);
  }

  // member: indices
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.indices.size() == 0) {
      out << "indices: []\n";
    } else {
      out << "indices:\n";
      for (auto item : msg.indices) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CloudIndexed & msg, bool use_flow_style = false)
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
  const gpd_ros2_interfaces::msg::CloudIndexed & msg,
  std::ostream & out, size_t indentation = 0)
{
  gpd_ros2_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gpd_ros2_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const gpd_ros2_interfaces::msg::CloudIndexed & msg)
{
  return gpd_ros2_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<gpd_ros2_interfaces::msg::CloudIndexed>()
{
  return "gpd_ros2_interfaces::msg::CloudIndexed";
}

template<>
inline const char * name<gpd_ros2_interfaces::msg::CloudIndexed>()
{
  return "gpd_ros2_interfaces/msg/CloudIndexed";
}

template<>
struct has_fixed_size<gpd_ros2_interfaces::msg::CloudIndexed>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<gpd_ros2_interfaces::msg::CloudIndexed>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<gpd_ros2_interfaces::msg::CloudIndexed>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_INDEXED__TRAITS_HPP_
