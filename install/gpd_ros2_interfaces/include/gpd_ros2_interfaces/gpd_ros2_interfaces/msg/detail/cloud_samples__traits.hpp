// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from gpd_ros2_interfaces:msg/CloudSamples.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SAMPLES__TRAITS_HPP_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SAMPLES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "gpd_ros2_interfaces/msg/detail/cloud_samples__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'cloud_sources'
#include "gpd_ros2_interfaces/msg/detail/cloud_sources__traits.hpp"
// Member 'samples'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace gpd_ros2_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const CloudSamples & msg,
  std::ostream & out)
{
  out << "{";
  // member: cloud_sources
  {
    out << "cloud_sources: ";
    to_flow_style_yaml(msg.cloud_sources, out);
    out << ", ";
  }

  // member: samples
  {
    if (msg.samples.size() == 0) {
      out << "samples: []";
    } else {
      out << "samples: [";
      size_t pending_items = msg.samples.size();
      for (auto item : msg.samples) {
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
  const CloudSamples & msg,
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

  // member: samples
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.samples.size() == 0) {
      out << "samples: []\n";
    } else {
      out << "samples:\n";
      for (auto item : msg.samples) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CloudSamples & msg, bool use_flow_style = false)
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
  const gpd_ros2_interfaces::msg::CloudSamples & msg,
  std::ostream & out, size_t indentation = 0)
{
  gpd_ros2_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gpd_ros2_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const gpd_ros2_interfaces::msg::CloudSamples & msg)
{
  return gpd_ros2_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<gpd_ros2_interfaces::msg::CloudSamples>()
{
  return "gpd_ros2_interfaces::msg::CloudSamples";
}

template<>
inline const char * name<gpd_ros2_interfaces::msg::CloudSamples>()
{
  return "gpd_ros2_interfaces/msg/CloudSamples";
}

template<>
struct has_fixed_size<gpd_ros2_interfaces::msg::CloudSamples>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<gpd_ros2_interfaces::msg::CloudSamples>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<gpd_ros2_interfaces::msg::CloudSamples>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__CLOUD_SAMPLES__TRAITS_HPP_
