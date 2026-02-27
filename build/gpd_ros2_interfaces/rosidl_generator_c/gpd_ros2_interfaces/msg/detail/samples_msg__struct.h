// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from gpd_ros2_interfaces:msg/SamplesMsg.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__MSG__DETAIL__SAMPLES_MSG__STRUCT_H_
#define GPD_ROS2_INTERFACES__MSG__DETAIL__SAMPLES_MSG__STRUCT_H_

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
// Member 'samples'
#include "geometry_msgs/msg/detail/point__struct.h"

/// Struct defined in msg/SamplesMsg in the package gpd_ros2_interfaces.
/**
  * This message describes a set of point samples at which to detect grasps.
 */
typedef struct gpd_ros2_interfaces__msg__SamplesMsg
{
  /// Header
  std_msgs__msg__Header header;
  /// The samples, as (x,y,z) points, at which to search for grasp candidates.
  geometry_msgs__msg__Point__Sequence samples;
} gpd_ros2_interfaces__msg__SamplesMsg;

// Struct for a sequence of gpd_ros2_interfaces__msg__SamplesMsg.
typedef struct gpd_ros2_interfaces__msg__SamplesMsg__Sequence
{
  gpd_ros2_interfaces__msg__SamplesMsg * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} gpd_ros2_interfaces__msg__SamplesMsg__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // GPD_ROS2_INTERFACES__MSG__DETAIL__SAMPLES_MSG__STRUCT_H_
