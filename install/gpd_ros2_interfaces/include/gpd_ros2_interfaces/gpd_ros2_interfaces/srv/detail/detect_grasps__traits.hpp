// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__TRAITS_HPP_
#define GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "gpd_ros2_interfaces/srv/detail/detect_grasps__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'cloud_indexed'
#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__traits.hpp"

namespace gpd_ros2_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const DetectGrasps_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: cloud_indexed
  {
    out << "cloud_indexed: ";
    to_flow_style_yaml(msg.cloud_indexed, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DetectGrasps_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: cloud_indexed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cloud_indexed:\n";
    to_block_style_yaml(msg.cloud_indexed, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DetectGrasps_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace gpd_ros2_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use gpd_ros2_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const gpd_ros2_interfaces::srv::DetectGrasps_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  gpd_ros2_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gpd_ros2_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const gpd_ros2_interfaces::srv::DetectGrasps_Request & msg)
{
  return gpd_ros2_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<gpd_ros2_interfaces::srv::DetectGrasps_Request>()
{
  return "gpd_ros2_interfaces::srv::DetectGrasps_Request";
}

template<>
inline const char * name<gpd_ros2_interfaces::srv::DetectGrasps_Request>()
{
  return "gpd_ros2_interfaces/srv/DetectGrasps_Request";
}

template<>
struct has_fixed_size<gpd_ros2_interfaces::srv::DetectGrasps_Request>
  : std::integral_constant<bool, has_fixed_size<gpd_ros2_interfaces::msg::CloudIndexed>::value> {};

template<>
struct has_bounded_size<gpd_ros2_interfaces::srv::DetectGrasps_Request>
  : std::integral_constant<bool, has_bounded_size<gpd_ros2_interfaces::msg::CloudIndexed>::value> {};

template<>
struct is_message<gpd_ros2_interfaces::srv::DetectGrasps_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'grasp_configs'
#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__traits.hpp"

namespace gpd_ros2_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const DetectGrasps_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: grasp_configs
  {
    out << "grasp_configs: ";
    to_flow_style_yaml(msg.grasp_configs, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DetectGrasps_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: grasp_configs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "grasp_configs:\n";
    to_block_style_yaml(msg.grasp_configs, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DetectGrasps_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace gpd_ros2_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use gpd_ros2_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const gpd_ros2_interfaces::srv::DetectGrasps_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  gpd_ros2_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gpd_ros2_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const gpd_ros2_interfaces::srv::DetectGrasps_Response & msg)
{
  return gpd_ros2_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<gpd_ros2_interfaces::srv::DetectGrasps_Response>()
{
  return "gpd_ros2_interfaces::srv::DetectGrasps_Response";
}

template<>
inline const char * name<gpd_ros2_interfaces::srv::DetectGrasps_Response>()
{
  return "gpd_ros2_interfaces/srv/DetectGrasps_Response";
}

template<>
struct has_fixed_size<gpd_ros2_interfaces::srv::DetectGrasps_Response>
  : std::integral_constant<bool, has_fixed_size<gpd_ros2_interfaces::msg::GraspConfigList>::value> {};

template<>
struct has_bounded_size<gpd_ros2_interfaces::srv::DetectGrasps_Response>
  : std::integral_constant<bool, has_bounded_size<gpd_ros2_interfaces::msg::GraspConfigList>::value> {};

template<>
struct is_message<gpd_ros2_interfaces::srv::DetectGrasps_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<gpd_ros2_interfaces::srv::DetectGrasps>()
{
  return "gpd_ros2_interfaces::srv::DetectGrasps";
}

template<>
inline const char * name<gpd_ros2_interfaces::srv::DetectGrasps>()
{
  return "gpd_ros2_interfaces/srv/DetectGrasps";
}

template<>
struct has_fixed_size<gpd_ros2_interfaces::srv::DetectGrasps>
  : std::integral_constant<
    bool,
    has_fixed_size<gpd_ros2_interfaces::srv::DetectGrasps_Request>::value &&
    has_fixed_size<gpd_ros2_interfaces::srv::DetectGrasps_Response>::value
  >
{
};

template<>
struct has_bounded_size<gpd_ros2_interfaces::srv::DetectGrasps>
  : std::integral_constant<
    bool,
    has_bounded_size<gpd_ros2_interfaces::srv::DetectGrasps_Request>::value &&
    has_bounded_size<gpd_ros2_interfaces::srv::DetectGrasps_Response>::value
  >
{
};

template<>
struct is_service<gpd_ros2_interfaces::srv::DetectGrasps>
  : std::true_type
{
};

template<>
struct is_service_request<gpd_ros2_interfaces::srv::DetectGrasps_Request>
  : std::true_type
{
};

template<>
struct is_service_response<gpd_ros2_interfaces::srv::DetectGrasps_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__TRAITS_HPP_
