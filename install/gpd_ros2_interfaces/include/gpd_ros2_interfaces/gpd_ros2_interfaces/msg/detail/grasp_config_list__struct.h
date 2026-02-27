// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from gpd_ros2_interfaces:msg/GraspConfigList.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__STRUCT_H_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'grasps'
#include "gpd_ros2_interfaces/msg/detail/grasp_config__struct.h"

/// Struct defined in msg/GraspConfigList in the package gpd_ros2_interfaces.
/**
  * This message stores a list of grasp configurations.
 */
typedef struct gpd_ros2_interfaces__msg__GraspConfigList
{
  /// The time of acquisition, and the coordinate frame ID.
  std_msgs__msg__Header header;
  /// The list of grasp configurations.
  gpd_ros2_interfaces__msg__GraspConfig__Sequence grasps;
} gpd_ros2_interfaces__msg__GraspConfigList;

// Struct for a sequence of gpd_ros2_interfaces__msg__GraspConfigList.
typedef struct gpd_ros2_interfaces__msg__GraspConfigList__Sequence
{
  gpd_ros2_interfaces__msg__GraspConfigList * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} gpd_ros2_interfaces__msg__GraspConfigList__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__GRASP_CONFIG_LIST__STRUCT_H_
