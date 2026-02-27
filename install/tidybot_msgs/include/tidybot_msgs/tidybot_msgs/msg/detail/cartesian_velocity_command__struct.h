// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tidybot_msgs:msg/CartesianVelocityCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__STRUCT_H_
#define TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__STRUCT_H_

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
// Member 'twist'
#include "geometry_msgs/msg/detail/twist__struct.h"
// Member 'frame_id'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/CartesianVelocityCommand in the package tidybot_msgs.
/**
  * End-effector Cartesian velocity command
  * Controls the velocity of the arm's end-effector in task space
 */
typedef struct tidybot_msgs__msg__CartesianVelocityCommand
{
  std_msgs__msg__Header header;
  /// Linear velocity (m/s) and angular velocity (rad/s)
  geometry_msgs__msg__Twist twist;
  /// Reference frame for the velocity command
  /// Options: "base_link" (world-aligned) or "end_effector" (tool-aligned)
  rosidl_runtime_c__String frame_id;
  /// Duration to apply velocity in seconds (0 = until next command)
  double duration;
} tidybot_msgs__msg__CartesianVelocityCommand;

// Struct for a sequence of tidybot_msgs__msg__CartesianVelocityCommand.
typedef struct tidybot_msgs__msg__CartesianVelocityCommand__Sequence
{
  tidybot_msgs__msg__CartesianVelocityCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tidybot_msgs__msg__CartesianVelocityCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__STRUCT_H_
