// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from gpd_ros2_interfaces:srv/DetectGrasps.idl
// generated code does not contain a copyright notice

#ifndef GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__STRUCT_H_
#define GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'cloud_indexed'
#include "gpd_ros2_interfaces/msg/detail/cloud_indexed__struct.h"

/// Struct defined in srv/DetectGrasps in the package gpd_ros2_interfaces.
typedef struct gpd_ros2_interfaces__srv__DetectGrasps_Request
{
  gpd_ros2_interfaces__msg__CloudIndexed cloud_indexed;
} gpd_ros2_interfaces__srv__DetectGrasps_Request;

// Struct for a sequence of gpd_ros2_interfaces__srv__DetectGrasps_Request.
typedef struct gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence
{
  gpd_ros2_interfaces__srv__DetectGrasps_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} gpd_ros2_interfaces__srv__DetectGrasps_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'grasp_configs'
#include "gpd_ros2_interfaces/msg/detail/grasp_config_list__struct.h"

/// Struct defined in srv/DetectGrasps in the package gpd_ros2_interfaces.
typedef struct gpd_ros2_interfaces__srv__DetectGrasps_Response
{
  gpd_ros2_interfaces__msg__GraspConfigList grasp_configs;
} gpd_ros2_interfaces__srv__DetectGrasps_Response;

// Struct for a sequence of gpd_ros2_interfaces__srv__DetectGrasps_Response.
typedef struct gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence
{
  gpd_ros2_interfaces__srv__DetectGrasps_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} gpd_ros2_interfaces__srv__DetectGrasps_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // GPD_ROS2_INTERFACES__SRV__DETAIL__DETECT_GRASPS__STRUCT_H_
